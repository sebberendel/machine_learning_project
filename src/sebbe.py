import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, make_scorer, recall_score
import joblib

from data_preprocessing import load_training_data, load_test_data, get_pipeline
# ----------------------------------------------------
# Load data
# ----------------------------------------------------

X_train, X_holdout, y_train, y_holdout = load_training_data()
X_test = load_test_data()

# ----------------------------------------------------
# RandomSearchCV
# ----------------------------------------------------
recall_high = make_scorer(
    recall_score,
    pos_label="high_bike_demand"
)

base_model = RandomForestClassifier(
    random_state=1,
    n_jobs=-1
)

base_pipeline = get_pipeline(base_model)

param_grid = {

    # Antal träd
    "model__n_estimators": [300],

    # Trädens djup
    "model__max_depth": [None, 10, 20, 30, 40],

    # Min split
    "model__min_samples_split": [2, 5, 10, 20],

    # Min leaf
    "model__min_samples_leaf": [1, 2, 5, 10],

    # Max features
    "model__max_features": ["sqrt", "log2", 0.3, 0.5],

    # 🔥 Klassvikter (viktigast!)
    "model__class_weight": [
        None,
        "balanced",
        {"high_bike_demand": 3, "low_bike_demand": 1},
        {"high_bike_demand": 5, "low_bike_demand": 1},
        {"high_bike_demand": 10, "low_bike_demand": 1},
        {"high_bike_demand": 15, "low_bike_demand": 1},
        {"high_bike_demand": 20, "low_bike_demand": 1},
        {"high_bike_demand": 30, "low_bike_demand": 1},
    ]
}

search = RandomizedSearchCV(
    base_pipeline,
    param_distributions=param_grid,
    n_iter=60,
    scoring=recall_high,
    cv=5,
    n_jobs=-1,
    random_state=1,
    verbose=1
)
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)

# ----------------------------------------------------
# Final model
# ----------------------------------------------------

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=40,
    max_features=0.5,
    class_weight={"high_bike_demand": 30, "low_bike_demand": 1},
    random_state=1,
    n_jobs=-1,
    min_samples_leaf=10,
    min_samples_split=2,
)

# ----------------------------------------------------
# Pipeline
# ----------------------------------------------------

pipeline = get_pipeline(final_model)

# ----------------------------------------------------
# Cross-validation
# ----------------------------------------------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

y_pred = cross_val_predict(
    pipeline,
    X_train, 
    y_train,
    cv=cv,
    n_jobs=-1
)

# ----------------------------------------------------
# Evaluation on training data
# ----------------------------------------------------

print(classification_report(y_train, y_pred))

ConfusionMatrixDisplay.from_predictions(y_train, y_pred)
plt.show()

