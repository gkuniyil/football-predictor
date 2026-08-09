"""
given current standings + remaining fixtures for a season, use the
trained classifier's probabilities to simulate the rest of the season
thousands of times, and report final-position probabilities per team.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os
import xgboost as xgb

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

STARTING_RATING = 1500
N_SIMULATIONS = 10000


# ---------------------------------------------------------------------------
# FEATURE FUNCTIONS (identical to features.py -- reused here so we can
# compute features for REMAINING fixtures too, not just historical ones)
# ---------------------------------------------------------------------------
def get_team_rating_as_of(team_id, date):
    query = text("""
        SELECT elo_rating FROM team_ratings
        WHERE team_id = :team_id AND date < :date
        ORDER BY date DESC LIMIT 1
    """)
    result = pd.read_sql(query, engine, params={"team_id": team_id, "date": date})
    if result.empty:
        return STARTING_RATING
    return result.iloc[0]["elo_rating"]


def get_recent_form(team_id, date, num_matches=5):
    query = text("""
        SELECT date, home_team_id, away_team_id, result
        FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND date < :date
        ORDER BY date DESC LIMIT :num_matches
    """)
    result = pd.read_sql(query, engine, params={
        "team_id": team_id, "date": date, "num_matches": num_matches
    })
    if result.empty:
        return 0
    total_points = 0
    for _, row in result.iterrows():
        if row["home_team_id"] == team_id:
            if row["result"] == "H":
                total_points += 3
            elif row["result"] == "D":
                total_points += 1
        else:
            if row["result"] == "A":
                total_points += 3
            elif row["result"] == "D":
                total_points += 1
    return total_points


def get_home_away_win_rate(team_id, date, is_home):
    if is_home:
        query = text("SELECT result FROM matches WHERE home_team_id = :team_id AND date < :date")
    else:
        query = text("SELECT result FROM matches WHERE away_team_id = :team_id AND date < :date")
    result = pd.read_sql(query, engine, params={"team_id": team_id, "date": date})
    if result.empty:
        return 0.5
    total_games = len(result)
    wins = (result["result"] == ("H" if is_home else "A")).sum()
    return wins / total_games


def get_rest_days(team_id, date, default_rest=7):
    query = text("""
        SELECT date FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND date < :date
        ORDER BY date DESC LIMIT 1
    """)
    result = pd.read_sql(query, engine, params={"team_id": team_id, "date": date})
    if result.empty:
        return default_rest
    previous_match_date = result.iloc[0]["date"]
    return (date - previous_match_date).days


def build_features_for_match(home_id, away_id, match_date):
    """Builds one feature row for a single hypothetical/future match."""
    return {
        "home_elo": get_team_rating_as_of(home_id, match_date),
        "away_elo": get_team_rating_as_of(away_id, match_date),
        "home_form": get_recent_form(home_id, match_date),
        "away_form": get_recent_form(away_id, match_date),
        "home_win_rate": get_home_away_win_rate(home_id, match_date, is_home=True),
        "away_win_rate": get_home_away_win_rate(away_id, match_date, is_home=False),
        "home_rest_days": get_rest_days(home_id, match_date),
        "away_rest_days": get_rest_days(away_id, match_date),
    }


# ---------------------------------------------------------------------------
# STEP 1: simulate ONE match's outcome given probabilities
# ---------------------------------------------------------------------------
def simulate_match(probs):
    """probs: [prob_H, prob_D, prob_A]. Returns one random outcome, weighted."""
    return np.random.choice(["H", "D", "A"], p=probs)


# ---------------------------------------------------------------------------
# STEP 2: simulate ONE full run of the remaining season
# ---------------------------------------------------------------------------
def simulate_one_season(remaining_fixtures, current_standings, model, feature_columns):
    """
    remaining_fixtures: DataFrame with home_team_id, away_team_id, home_name,
                         away_name, and pre-computed feature columns.
    current_standings: dict {team_name: current_points}
    Returns: dict {team_name: final_points}
    """
    standings = current_standings.copy()  # don't mutate the original

    for _, match in remaining_fixtures.iterrows():
        X_match = match[feature_columns].values.reshape(1, -1)
        probs = model.predict_proba(X_match)[0]  # [prob_H, prob_D, prob_A]
        outcome = simulate_match(probs)

        home_name = match["home_name"]
        away_name = match["away_name"]

        if outcome == "H":
            standings[home_name] += 3
        elif outcome == "A":
            standings[away_name] += 3
        else:
            standings[home_name] += 1
            standings[away_name] += 1

    return standings


# ---------------------------------------------------------------------------
# STEP 3: run N_SIMULATIONS full seasons, aggregate into position probabilities
# ---------------------------------------------------------------------------
def run_monte_carlo(remaining_fixtures, current_standings, model, feature_columns, n_simulations=N_SIMULATIONS):
    all_results = []  # will hold one final-standings dict per simulation

    for i in range(n_simulations):
        final_standings = simulate_one_season(remaining_fixtures, current_standings, model, feature_columns)
        all_results.append(final_standings)

    results_df = pd.DataFrame(all_results)  # one row per simulation, one column per team

    # For each team, figure out what RANK they finished at, in each simulation
    ranks_df = results_df.rank(axis=1, ascending=False, method="min")

    summary = pd.DataFrame({
        "avg_points": results_df.mean(),
        "title_probability": (ranks_df == 1).mean(),
        "top4_probability": (ranks_df <= 4).mean(),
    }).sort_values("title_probability", ascending=False)

    return summary


print("simulate_season.py loaded. Functions ready -- next: build current_standings and remaining_fixtures.")


# ---------------------------------------------------------------------------
# STEP 4: train the model (same logic as train_model.py, reused here so
# this file can run standalone)
# ---------------------------------------------------------------------------
def train_classifier():
    df = pd.read_csv("data/match_features.csv")
    feature_columns = [
        "home_elo", "away_elo", "home_form", "away_form",
        "home_win_rate", "away_win_rate", "home_rest_days", "away_rest_days"
    ]
    X = df[feature_columns]
    result_map = {"H": 0, "D": 1, "A": 2}
    y = df["result"].map(result_map)

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3,
        max_depth=3, n_estimators=100, learning_rate=0.05,
        reg_lambda=1.0, subsample=0.8, colsample_bytree=0.8,
    )
    model.fit(X, y)  # train on ALL historical data this time (not just 80%),
                      # since we're using this for real simulation, not evaluation
    return model, feature_columns


# ---------------------------------------------------------------------------
# STEP 5: pick a cutoff date, compute REAL standings as of that date,
# pull REAL remaining fixtures, build features for them
# ---------------------------------------------------------------------------
def get_standings_as_of(cutoff_date):
    """Computes real league standings using only matches before cutoff_date."""
    query = text("""
        SELECT m.home_team_id, m.away_team_id, m.result,
               th.name AS home_name, ta.name AS away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.date < :cutoff AND m.competition = 'la_liga'
    """)
    played = pd.read_sql(query, engine, params={"cutoff": cutoff_date})

    standings = {}
    for _, row in played.iterrows():
        standings.setdefault(row["home_name"], 0)
        standings.setdefault(row["away_name"], 0)
        if row["result"] == "H":
            standings[row["home_name"]] += 3
        elif row["result"] == "A":
            standings[row["away_name"]] += 3
        else:
            standings[row["home_name"]] += 1
            standings[row["away_name"]] += 1

    return standings


def get_remaining_fixtures(cutoff_date, feature_columns):
    """Pulls real fixtures after cutoff_date, builds features for each."""
    query = text("""
        SELECT m.match_id, m.date, m.home_team_id, m.away_team_id,
               th.name AS home_name, ta.name AS away_name
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.date >= :cutoff
        ORDER BY m.date
    """)
    fixtures = pd.read_sql(query, engine, params={"cutoff": cutoff_date})

    rows = []
    for _, row in fixtures.iterrows():
        feats = build_features_for_match(row["home_team_id"], row["away_team_id"], row["date"])
        feats["home_name"] = row["home_name"]
        feats["away_name"] = row["away_name"]
        rows.append(feats)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RUN IT
# ---------------------------------------------------------------------------
def get_season_champions():
    """
    Computes the top-of-table team for each of the 5 seasons in the dataset,
    by summing points across all matches in each season's date range.
    Note: this uses raw points only, no head-to-head tiebreaker logic --
    a known simplification, flagged here rather than silently assumed correct.
    """
    seasons = [
        ("2021-22", "2021-08-01", "2022-07-01"),
        ("2022-23", "2022-08-01", "2023-07-01"),
        ("2023-24", "2023-08-01", "2024-07-01"),
        ("2024-25", "2024-08-01", "2025-07-01"),
        ("2025-26", "2025-08-01", "2026-07-01"),
    ]
    results = []
    for label, start, end in seasons:
        query = text("""
            SELECT m.home_team_id, m.away_team_id, m.result,
                   th.name AS home_name, ta.name AS away_name
            FROM matches m
            JOIN teams th ON m.home_team_id = th.team_id
            JOIN teams ta ON m.away_team_id = ta.team_id
            WHERE m.date >= :start AND m.date < :end
        """)
        season_matches = pd.read_sql(query, engine, params={"start": start, "end": end})
        standings = {}
        for _, row in season_matches.iterrows():
            standings.setdefault(row["home_name"], 0)
            standings.setdefault(row["away_name"], 0)
            if row["result"] == "H":
                standings[row["home_name"]] += 3
            elif row["result"] == "A":
                standings[row["away_name"]] += 3
            else:
                standings[row["home_name"]] += 1
                standings[row["away_name"]] += 1
        if standings:
            champion = max(standings, key=standings.get)
            results.append({"season": label, "champion": champion, "points": standings[champion]})
    return results


def get_recent_matches(before_date, limit=20):
    """Returns the most recent real matches before a given date, most recent first."""
    query = text("""
        SELECT m.date, th.name AS home_name, ta.name AS away_name,
               m.home_goals, m.away_goals, m.result
        FROM matches m
        JOIN teams th ON m.home_team_id = th.team_id
        JOIN teams ta ON m.away_team_id = ta.team_id
        WHERE m.date < :before_date
        ORDER BY m.date DESC
        LIMIT :limit
    """)
    return pd.read_sql(query, engine, params={"before_date": before_date, "limit": limit})


if __name__ == "__main__":
    CUTOFF_DATE = "2026-02-01"  # roughly 3/4 through the 2025-26 season

    print("Training classifier on full historical data...")
    model, feature_columns = train_classifier()

    print(f"Computing real standings as of {CUTOFF_DATE}...")
    current_standings = get_standings_as_of(CUTOFF_DATE)
    print(current_standings)

    print("Building features for remaining fixtures...")
    remaining_fixtures = get_remaining_fixtures(CUTOFF_DATE, feature_columns)
    print(f"{len(remaining_fixtures)} remaining fixtures found")

    print(f"Running {N_SIMULATIONS} simulations...")
    summary = run_monte_carlo(remaining_fixtures, current_standings, model, feature_columns)
    print(summary)