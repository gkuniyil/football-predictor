# La Liga Match Predictor

An end-to-end machine learning system that predicts La Liga match outcomes and simulates full-season standings, built on 5 seasons of real historical data.

**Live API:** https://football-predictor-production-f666.up.railway.app/docs

## What it does

Given two teams and a date, the model predicts win/draw/loss probabilities using Elo ratings, recent form, home/away splits, and rest days as features. A Monte Carlo simulator then replays a league's remaining fixtures thousands of times to project title and top-4 probabilities for every team.

## How it works

Data ingestion pulls 5 seasons of La Liga results (1,900 matches) into PostgreSQL. An Elo rating engine walks through every match chronologically, updating team strength ratings with a home-advantage adjustment. Feature engineering then builds a leakage-safe training table — every feature for a given match uses only information available strictly before that match's date, verified with automated pytest checks.

An XGBoost classifier is trained on this feature table and evaluated against a frequency-based naive baseline (not a strawman): the model beat the baseline on log loss (0.97 vs 1.05) and Brier score, after diagnosing and fixing an initial overfitting issue through regularization tuning. Predictions were also validated against a real historical title race, correctly favoring the actual champion.

A Monte Carlo simulator uses the trained classifier's per-match probabilities to run thousands of simulated replays of a season's remaining fixtures, aggregating results into title and top-4 probability estimates per team.

The whole pipeline is served through a FastAPI backend, with Redis caching on the slowest endpoint (season simulation) so repeated requests return instantly instead of recomputing. Experiments are tracked with MLflow. An interactive React frontend lets you pick any two teams and a date to see live predictions, browse real match history, and view projected standings.

## Stack

Python, PostgreSQL, Redis, XGBoost, FastAPI, React, MLflow, Docker, Prometheus, Grafana, deployed on Railway.

## Project structure

```
src/
  ingest.py           historical data ingestion
  elo_engine.py        Elo rating engine
  features.py           leakage-safe feature engineering
  train_model.py         classifier training and evaluation
  simulate_season.py     Monte Carlo season simulator
  api.py                  FastAPI serving layer
  drift_check.py           model drift monitoring
frontend/
  index.html               React frontend
tests/
  test_correctness.py         pytest correctness checks
```

## Running locally

```bash
docker compose up -d --build
```

This starts PostgreSQL, Redis, the API, Prometheus, and Grafana. Run the pipeline scripts once to populate the database:

```bash
python src/ingest.py
python src/elo_engine.py
python src/features.py
```

Then visit `localhost:8000/docs` for the API and serve `frontend/index.html` to see the dashboard.

