"""
Run with: uvicorn src.api:app --reload
Then visit http://localhost:8000/docs for interactive API docs.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
from datetime import datetime

from src.simulate_season import (
    build_features_for_match, train_classifier,
    get_standings_as_of, get_remaining_fixtures, run_monte_carlo,
    get_season_champions, get_recent_matches, engine
)
from sqlalchemy import text as sql_text

app = FastAPI(title="La Liga Match Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Train once at startup, reuse across requests (don't retrain per request)
model, feature_columns = train_classifier()


@app.get("/teams")
def list_teams():
    """Returns every team's id and name, sorted alphabetically -- powers the dropdowns."""
    df = pd.read_sql(sql_text("SELECT team_id, name FROM teams ORDER BY name"), engine)
    return df.to_dict(orient="records")


@app.get("/champions")
def champions():
    """Returns the top-of-table team for each of the last 5 seasons."""
    return get_season_champions()


@app.get("/matches")
def matches(before_date: str, limit: int = 20):
    """Returns real match results before a given date, most recent first."""
    df = get_recent_matches(before_date, limit)
    df["date"] = df["date"].astype(str)
    return df.to_dict(orient="records")


@app.get("/predict")
def predict_match(home_team_id: int, away_team_id: int, date: str):
    """Predict H/D/A probabilities for a specific matchup on a given date."""
    parsed_date = datetime.strptime(date, "%Y-%m-%d").date()
    feats = build_features_for_match(home_team_id, away_team_id, parsed_date)
    X = pd.DataFrame([feats])[feature_columns]
    probs = model.predict_proba(X)[0]
    return {"home_win": float(probs[0]), "draw": float(probs[1]), "away_win": float(probs[2])}


@app.get("/simulate-season")
def simulate_season_endpoint(cutoff_date: str = "2026-02-01", n_simulations: int = 10000):
    """Run the Monte Carlo season simulation from a given cutoff date."""
    current_standings = get_standings_as_of(cutoff_date)
    remaining_fixtures = get_remaining_fixtures(cutoff_date, feature_columns)
    summary = run_monte_carlo(remaining_fixtures, current_standings, model, feature_columns, n_simulations)
    return summary.to_dict(orient="index")