"""
Day 6-7: Train the XGBoost classifier
Purpose: load match_features.csv, split by TIME (not randomly), train an
XGBoost multi-class classifier, evaluate with log loss + Brier score.
"""

import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
import xgboost as xgb

# load the feature table
df = pd.read_csv("data/match_features.csv")
print(df.shape)          # sanity check -- should be (1900, 11)
print(df.head())         # eyeball a few rows

# separate features (X) from the label (y).
feature_columns = [
    "home_elo", "away_elo", "home_form", "away_form",
    "home_win_rate", "away_win_rate", "home_rest_days", "away_rest_days"
]

X = df[feature_columns]
y = df["result"]


#XGBoost needs numeric labels so map them on to 0,1,2
result_map = {"H": 0, "D": 1, "A": 2}
y_numeric = y.map(result_map)

# slice the first 80% for training vs last 20% for testing 
split_index = int(len(df) * 0.8)  # 80% cutoff point, as a row index 
X_train = X[:split_index]   # features for rows BEFORE the cutoff 
X_test = X[split_index:]    # features for rows FROM the cutoff onward

y_train = y_numeric[:split_index]  # labels matching X_train, same row range
y_test = y_numeric[split_index:]   # labels matching X_test, same row range

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")  # sanity check -- should be ~1520 / ~380

#test and train model using xgb (3 classes are H, D, A)
# STEP 5: train the model, with settings tuned to avoid overfitting
# on a relatively small dataset (1,520 training rows).
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    max_depth=3,
    n_estimators=100,
    learning_rate=0.05,
    reg_lambda=1.0,
    subsample=0.8,
    colsample_bytree=0.8,
)
model.fit(X_train, y_train)
print("Model trained successfully.")


# STEP 6: predicted probabilities on the test set
predicted_probs = model.predict_proba(X_test)
print(predicted_probs[:5])


# STEP 7: log loss
loss = log_loss(y_test, predicted_probs)
print(f"Log loss: {loss:.4f}")


# STEP 8: accuracy
predicted_labels = model.predict(X_test)
accuracy = (predicted_labels == y_test).mean()
print(f"Accuracy: {accuracy:.4f}")


# STEP 9: naive baseline comparison -- always predict "Home win"
import numpy as np
naive_probs = np.tile([1.0, 0.0, 0.0], (len(y_test), 1))
naive_loss = log_loss(y_test, naive_probs, labels=[0, 1, 2])
print(f"Naive baseline log loss: {naive_loss:.4f}")

# PIECE 1: fair naive baseline -- predict the real historical H/D/A frequencies
# from training data, instead of 100% certainty on one outcome.

class_frequencies = y_train.value_counts(normalize=True).sort_index()
print(class_frequencies)
# Counts how often 0/1/2 (H/D/A) actually appeared in the 1520 TRAINING matches,
# converted to percentages that sum to 1.

naive_probs_v2 = np.tile(class_frequencies.values, (len(y_test), 1))
# Takes those 3 percentages and repeats them 380 times -- once for each of
# the 380 TEST matches -- since this baseline guesses the same thing every time.

naive_loss_v2 = log_loss(y_test, naive_probs_v2, labels=[0, 1, 2])
print(f"Frequency-based naive baseline log loss: {naive_loss_v2:.4f}")
# Scores this honest-but-uninformed guessing strategy using the same metric
# (log loss) we used to score your real model, so the two are directly comparable.


from sklearn.preprocessing import label_binarize

y_test_binarized = label_binarize(y_test, classes=[0, 1, 2])

brier_scores = []
for i in range(3):
    score = brier_score_loss(y_test_binarized[:, i], predicted_probs[:, i])
    brier_scores.append(score)

avg_brier = sum(brier_scores) / len(brier_scores)
print(f"Brier scores by class (H, D, A): {brier_scores}")
print(f"Average Brier score: {avg_brier:.4f}")