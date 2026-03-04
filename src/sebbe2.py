import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import joblib

from data_preprocessing import load_training_data, load_test_data, get_pipeline
# Ladda data
X_train, X_holdout, y_train, y_holdout = load_training_data()
X_test = load_test_data()

# ----------------------------------------------------
# Final model
# ----------------------------------------------------

final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    max_features="sqrt",
    class_weight={"high_bike_demand": 16, "low_bike_demand": 1},
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
    X_train,   # OBS: rå data
    y_train,
    cv=cv,
    n_jobs=-1
)

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------

print(classification_report(y_train, y_pred))

ConfusionMatrixDisplay.from_predictions(y_train, y_pred)
plt.show()

# ----------------------------------------------------
# Fit final model on ALL training data
# ----------------------------------------------------

pipeline.fit(X_train, y_train)

# Spara hela pipelinen
joblib.dump(pipeline, "model.pkl")

# ----------------------------------------------------
# Utvärdera på training data
# ----------------------------------------------------
x_test_pred = pipeline.predict(X_test)

binary_predictions = np.where(
    x_test_pred == "high_bike_demand", 1, 0
)
# np.savetxt(
#     "predictions.csv",
#     [binary_predictions],
#     fmt="%d",
#     delimiter=","
# )
print(binary_predictions)