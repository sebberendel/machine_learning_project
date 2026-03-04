import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    recall_score,
    make_scorer
)

from data_preprocessing import load_training_data, load_test_data, get_pipeline


# ----------------------------------------------------
# Load data
# ----------------------------------------------------
X_train, X_holdout, y_train, y_holdout = load_training_data()
X_test = load_test_data()

# ----------------------------------------------------
# Cross-validation setup
# ----------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# ====================================================
# 1) BASELINE
# ====================================================
baseline_model = LogisticRegression(
    max_iter=5000,
    random_state=1,
    class_weight={"high_bike_demand": 1, "low_bike_demand": 1}
)

baseline_pipe = get_pipeline(baseline_model)

y_pred_base = cross_val_predict(
    baseline_pipe,
    X_train,   
    y_train,
    cv=cv,
    n_jobs=-1
)

print("=== BASELINE (otunad) ===")
print("Accuracy:", accuracy_score(y_train, y_pred_base))
print("Recall (high_bike_demand):", recall_score(y_train, y_pred_base, pos_label="high_bike_demand"))
print("\nClassification Report (Baseline):")
print(classification_report(y_train, y_pred_base, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_train, y_pred_base)
plt.title("Confusion Matrix - Baseline (CV on train)")
plt.show()


# ====================================================
# 2) GRID SEARCH (C + solver)
# ====================================================

recall_high = make_scorer(
    recall_score,
    pos_label="high_bike_demand"
)

tune_model = LogisticRegression(
    max_iter=5000,
    random_state=1,
    class_weight={"high_bike_demand": 10, "low_bike_demand": 1},
)

tune_pipe = get_pipeline(tune_model)

param_grid = {
    "model__C": np.logspace(-4, 2, 6),
    "model__solver": ["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
}

gs = GridSearchCV(
    tune_pipe,
    param_grid=param_grid,
    scoring="accuracy",    
    cv=cv,
    n_jobs=-1,
    error_score=0,
    refit=True,
)

gs.fit(X_train, y_train)

print("\n=== GRID SEARCH RESULT ===")
print("Best hyperparameters:", gs.best_params_)
print("Best CV-score (accuracy):", gs.best_score_)

best_pipe = gs.best_estimator_

# CV-predictions
y_pred_tuned = cross_val_predict(
    best_pipe,
    X_train,
    y_train,
    cv=cv,
    n_jobs=-1
)

print("\n=== TUNED (best C + solver) ===")
print("Accuracy:", accuracy_score(y_train, y_pred_tuned))
print("Recall (high_bike_demand):", recall_score(y_train, y_pred_tuned, pos_label="high_bike_demand"))
print("\nClassification Report (Tuned):")
print(classification_report(y_train, y_pred_tuned, zero_division=0))

ConfusionMatrixDisplay.from_predictions(y_train, y_pred_tuned)
plt.title("Confusion Matrix - Tuned (CV on train)")
plt.show()