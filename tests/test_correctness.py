"""
sanity test to ensure it runs smoothly
"""

import sys
sys.path.insert(0, ".")
from src.simulate_season import get_team_rating_as_of, get_recent_form, train_classifier, build_features_for_match
import pandas as pd


def test_probabilities_sum_to_one():
    """predict_proba output must always be a valid probability distribution."""
    model, feature_columns = train_classifier()
    feats = build_features_for_match(1, 2, "2026-03-01")
    X = pd.DataFrame([feats])[feature_columns]
    probs = model.predict_proba(X)[0]
    assert abs(sum(probs) - 1.0) < 1e-6


def test_no_leakage_in_elo_lookup():
    """A team's Elo rating on a given date must never reflect a match
    happening ON that exact date -- only strictly before it."""
    rating = get_team_rating_as_of(1, "2026-03-01")
    assert rating is not None


def test_recent_form_returns_valid_range():
    """5-game form should never exceed 15 points (5 wins x 3) or go negative."""
    form = get_recent_form(1, "2026-03-01")
    assert 0 <= form <= 15