# - Imports --------------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors as skl_nb
import sklearn.discriminant_analysis as skl_da
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

# - Define train/test data -----------------------------
from data_preprocessing import load_training_data, get_pipeline

np.random.seed(10)

X_train, X_holdout, y_train, y_holdout  = load_training_data()

# - Final models -----------------------------------------

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=40,
    max_features=0.5,
    class_weight={"high_bike_demand": 30, "low_bike_demand": 1},
    random_state=1,
    n_jobs=-1,
    min_samples_leaf=10,
    min_samples_split=2,
)

kNN = skl_nb.KNeighborsClassifier(n_neighbors=6)

# --

log_reg = LogisticRegression(
    class_weight={'high_bike_demand': 10, 'low_bike_demand': 1},
    random_state=1,
    max_iter=1000,
    C=0.4,
    solver='newton-cg',
)

# --

priors = [0.61, 0.39] 
QDA = skl_da.QuadraticDiscriminantAnalysis(reg_param=0.10526315789473684, priors=priors)

# Create a list of models to evaluate
models = [random_forest, kNN, log_reg, QDA]

predictions = {}
# Testing all models
for model in models:
    pipeline = get_pipeline(model)
    model.fit(X_train, y_train)
    y_holdout_pred = model.predict(X_holdout)
    predictions[str(model)] = y_holdout_pred
    print(model, "\n", classification_report(y_holdout, y_holdout_pred), "\n\n")


