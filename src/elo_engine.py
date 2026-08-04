import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

STARTING_RATING = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 75

def expected_score(rating_a, rating_b):
    """
    Returns the probability that team A beats team B, given their ratings.
    Formula: 1 / (1 + 10^((rating_b - rating_a) / 400))
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def update_ratings(rating_home, rating_away, result):
    """
    result: 'H' (home win), 'D' (draw), 'A' (away win)
    """
    temp_home_rating = rating_home + HOME_ADVANTAGE

    expected_home = expected_score(temp_home_rating, rating_away)
    expected_away = 1 - expected_home

    if result == 'H':
        actual_home = 1
        actual_away = 0
    elif result == 'A':
        actual_home = 0
        actual_away = 1
    else:
        actual_home = 0.5
        actual_away = 0.5

    new_rating_home = rating_home + K_FACTOR * (actual_home - expected_home)
    new_rating_away = rating_away + K_FACTOR * (actual_away - expected_away)

    return new_rating_home, new_rating_away


matches_df = pd.read_sql("""
    SELECT m.date, m.result, m.match_id,
           th.team_id AS home_team_id, th.name AS home_team_name,
           ta.team_id AS away_team_id, ta.name AS away_team_name
    FROM matches m
    JOIN teams th ON m.home_team_id = th.team_id
    JOIN teams ta ON m.away_team_id = ta.team_id
    ORDER BY m.date
""", engine)

all_teams = pd.concat([matches_df["home_team_name"], matches_df["away_team_name"]]).unique()
current_ratings = {team: STARTING_RATING for team in all_teams}

print(f"Initialized {len(current_ratings)} teams at {STARTING_RATING}")

rating_history_rows = []

for _, row in matches_df.iterrows():
    home_name = row["home_team_name"]
    away_name = row["away_team_name"]

    home_rating = current_ratings[home_name]
    away_rating = current_ratings[away_name]

    new_home_rating, new_away_rating = update_ratings(home_rating, away_rating, row["result"])

    current_ratings[home_name] = new_home_rating
    current_ratings[away_name] = new_away_rating

    rating_history_rows.append({"team_id": row["home_team_id"], "date": row["date"], "elo_rating": new_home_rating})
    rating_history_rows.append({"team_id": row["away_team_id"], "date": row["date"], "elo_rating": new_away_rating})

print(f"Processed all matches. Built {len(rating_history_rows)} rating history rows.")

history_df = pd.DataFrame(rating_history_rows)
with engine.begin() as conn:
    for _, r in history_df.iterrows():
        conn.execute(
            text("""
                INSERT INTO team_ratings (team_id, date, elo_rating)
                VALUES (:team_id, :date, :elo_rating)
                ON CONFLICT (team_id, date) DO UPDATE SET elo_rating = EXCLUDED.elo_rating
            """),
            {"team_id": int(r["team_id"]), "date": r["date"], "elo_rating": float(r["elo_rating"])}
        )

print("Done. Ratings saved to team_ratings.")