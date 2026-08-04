import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

STARTING_RATING = 1500


# Looks up a team's Elo rating as it stood right before a given match date.
# Falls back to STARTING_RATING (1500) if the team has no prior history yet.
def get_team_rating_as_of(team_id, date):
    query = text("""
        SELECT elo_rating FROM team_ratings
        WHERE team_id = :team_id
          AND date < :date
        ORDER BY date DESC
        LIMIT 1
    """)
    result = pd.read_sql(query, engine, params={"team_id": team_id, "date": date})

    if result.empty:
        return STARTING_RATING
    else:
        return result.iloc[0]["elo_rating"]


# Sums points (win=3, draw=1, loss=0) from a team's last 5 matches before
# a given date, checking both home and away matches. Captures short-term
# momentum that the slower-moving Elo rating doesn't fully reflect.
def get_recent_form(team_id, date, num_matches=5):
    query = text("""
        SELECT date, home_team_id, away_team_id, result
        FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND date < :date
        ORDER BY date DESC
        LIMIT :num_matches
    """)
    result = pd.read_sql(query, engine, params={
        "team_id": team_id, "date": date, "num_matches": num_matches
    })

    if result.empty:
        return 0

    total_points = 0
    for _, row in result.iterrows():
        # result is stored relative to home/away, so figure out
        # whether THIS team was home or away in each row first
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


# Computes a team's win rate specifically as the home side OR specifically
# as the away side (controlled by is_home), before a given date. Captures
# venue-specific tendencies separate from overall team strength.
def get_home_away_win_rate(team_id, date, is_home):
    if is_home:
        query = text("""
            SELECT result FROM matches
            WHERE home_team_id = :team_id AND date < :date
        """)
    else:
        query = text("""
            SELECT result FROM matches
            WHERE away_team_id = :team_id AND date < :date
        """)

    result = pd.read_sql(query, engine, params={"team_id": team_id, "date": date})

    if result.empty:
        return 0.5  # no data yet -- neutral default, not "definitely loses"

    total_games = len(result)
    if is_home:
        wins = (result["result"] == "H").sum()
    else:
        wins = (result["result"] == "A").sum()

    return wins / total_games


# Computes days since a team's most recent previous match (home or away).
# Acts as a fatigue proxy -- a signal none of the other features capture,
# since it's about scheduling/timing, not team quality.
def get_rest_days(team_id, date, default_rest=7):
    query = text("""
        SELECT date FROM matches
        WHERE (home_team_id = :team_id OR away_team_id = :team_id)
          AND date < :date
        ORDER BY date DESC
        LIMIT 1
    """)
    result = pd.read_sql(query, engine, params={"team_id": team_id, "date": date})

    if result.empty:
        return default_rest

    previous_match_date = result.iloc[0]["date"]
    gap = (date - previous_match_date).days
    return gap


# Main driver: loops through every historical match, calls all 4 feature
# functions for both the home and away team, and assembles one flat row
# per match. Saves the result as a CSV -- this becomes the training data
# for the Day 6 classifier.
def build_feature_table():
    matches_df = pd.read_sql("""
        SELECT match_id, date, home_team_id, away_team_id, result
        FROM matches
        ORDER BY date
    """, engine)

    feature_rows = []

    for _, row in matches_df.iterrows():
        match_date = row["date"]
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        home_elo = get_team_rating_as_of(home_id, match_date)
        away_elo = get_team_rating_as_of(away_id, match_date)

        home_form = get_recent_form(home_id, match_date)
        away_form = get_recent_form(away_id, match_date)

        home_win_rate = get_home_away_win_rate(home_id, match_date, is_home=True)
        away_win_rate = get_home_away_win_rate(away_id, match_date, is_home=False)

        home_rest = get_rest_days(home_id, match_date)
        away_rest = get_rest_days(away_id, match_date)

        feature_rows.append({
            "match_id": row["match_id"],
            "date": match_date,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_form": home_form,
            "away_form": away_form,
            "home_win_rate": home_win_rate,
            "away_win_rate": away_win_rate,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest,
            "result": row["result"],
        })

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv("data/match_features.csv", index=False)
    print(f"Built and saved {len(features_df)} feature rows to data/match_features.csv")
    return features_df


build_feature_table()Í