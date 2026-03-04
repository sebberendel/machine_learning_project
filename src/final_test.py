import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from data_preprocessing import load_training_data, load_test_data, get_pipeline

# ----------------------------------------------------
# Load data
# ----------------------------------------------------

X, y = load_training_data()
X_test = load_test_data()

# ----------------------------------------------------
# Final model
# ----------------------------------------------------



# ----------------------------------------------------
# Pipeline
# ----------------------------------------------------

pipeline = get_pipeline(final_model)

# ----------------------------------------------------
# Fit final model on ALL training data
# ----------------------------------------------------

pipeline.fit(X, y)

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