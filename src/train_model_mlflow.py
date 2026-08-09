"""
 logs hyperparameters and evaluation metrics for every run, so their is  queryable history of what you tried and what worked.
"""

import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import label_binarize
import xgboost as xgb
import mlflow
import numpy as np

mlflow.set_experiment("la_liga_match_predictor")

df = pd.read_csv("data/match_features.csv")
feature_columns = [
    "home_elo", "away_elo", "home_form", "away_form",
    "home_win_rate", "away_win_rate", "home_rest_days", "away_rest_days"
]
X = df[feature_columns]
result_map = {"H": 0, "D": 1, "A": 2}
y_numeric = df["result"].map(result_map)

split_index = int(len(df) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y_numeric[:split_index], y_numeric[split_index:]

params = {
    "objective": "multi:softprob", "num_class": 3,
    "max_depth": 2, "n_estimators": 100, "learning_rate": 0.05,
    "reg_lambda": 1.0, "subsample": 0.8, "colsample_bytree": 0.8,
}

with mlflow.start_run():
    mlflow.log_params(params)

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    predicted_probs = model.predict_proba(X_test)
    loss = log_loss(y_test, predicted_probs)
    accuracy = (model.predict(X_test) == y_test).mean()

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    brier = np.mean([
        brier_score_loss(y_test_bin[:, i], predicted_probs[:, i]) for i in range(3)
    ])

    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("brier_score", brier)
    mlflow.xgboost.log_model(model, "model")

    print(f"Logged run -- log_loss: {loss:.4f}, accuracy: {accuracy:.4f}, brier: {brier:.4f}")
    print("Run 'mlflow ui' in your terminal, then visit http://localhost:5000 to view this run.")