"""

Purpose: load la_liga_combined.csv into the Postgres teams/matches tables.

"""

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

load_dotenv()  # reads your .env file so DATABASE_URL is available

engine = create_engine(os.getenv("DATABASE_URL"))

# PART 1: Load the CSV and get unique team names
df = pd.read_csv("data/la_liga_combined.csv")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True).dt.strftime("%Y-%m-%d")

# Get a list (or set) of every unique team name across BOTH the HomeTeam and AwayTeam columns combined.
unique_teams = pd.concat([df["HomeTeam"], df["AwayTeam"]]).unique()

print(f"Found {len(unique_teams)} unique teams")  # sanity check -- should be ~20-25 for La Liga

# PART 2: Insert teams (skip duplicates)

# For each name in unique_teams, insert it into the `teams` table.
with engine.begin() as conn:
  for team_name in unique_teams:
    conn.execute(
             text("INSERT INTO teams (name) VALUES (:name) ON CONFLICT (name) DO NOTHING"),
               {"name": team_name}
          )
  print(f"Inserted teams (duplicates skipped automatically)")


# Build the team_name -> team_id lookup dictionary (ONE query)
teams_df = pd.read_sql("SELECT team_id, name FROM teams", engine)
team_lookup = dict(zip(teams_df["name"], teams_df["team_id"]))

print(f"Lookup dictionary has {len(team_lookup)} teams")



# PART 4: Insert matches
df = df.dropna(subset=["FTHG", "FTAG", "FTR"])
with engine.begin() as conn:
    for _, row in df.iterrows():
        conn.execute(
            text("""
                INSERT INTO matches (date, home_team_id, away_team_id, home_goals, away_goals, result, competition)
                VALUES (:date, :home_id, :away_id, :home_goals, :away_goals, :result, 'la_liga')
                ON CONFLICT (date, home_team_id, away_team_id) DO NOTHING
            """),
            {
                "date": row["Date"],
                "home_id": team_lookup[row["HomeTeam"]],
                "away_id": team_lookup[row["AwayTeam"]],
                "home_goals": row["FTHG"],
                "away_goals": row["FTAG"],
                "result": row["FTR"],
            }
        )

print("Done inserting matches.")