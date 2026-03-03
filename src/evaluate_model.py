# - Imports --------------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.neighbors as skl_nb
import sklearn.discriminant_analysis as skl_da
from sklearn.metrics import classification_report

# import sys

# sys.path.append('../src')  # lägg till src i sökvägarna

# - Define train/test data -----------------------------
from test_data_preprocessing import X, y, get_pipeline, X_holdout, y_holdout

np.random.seed(1)

# - Final models -----------------------------------------
from sklearn.linear_model import LogisticRegression

# --

random_forest = RandomForestClassifier(
    n_estimators=300,
    max_depth=20,
    max_features='sqrt',
    class_weight={'high_bike_demand': 16, 'low_bike_demand': 1},
    random_state=1,
    n_jobs=-1,
    min_samples_leaf=10,
    min_samples_split=2
)

kNN = skl_nb.KNeighborsClassifier(n_neighbors=6)

# --

log_reg = LogisticRegression(
    class_weight={'high_bike_demand': 10, 'low_bike_demand': 1},
    random_state=1,
    max_iter=1000,
    C=1.0,
    solver='liblinear',
)

# --

priors = [0.61, 0.39] 
LDA = skl_da.LinearDiscriminantAnalysis(priors=priors, tol=1e-4, )
QDA = skl_da.QuadraticDiscriminantAnalysis(reg_param=0.21052631578947367, priors=priors)

# Create a list of models to evaluate
models = [random_forest, kNN, log_reg, LDA, QDA]

# Testing all models
for model in models:
    pipeline = get_pipeline(model)
    model.fit(X, y)
    y_holdout_pred = model.predict(X_holdout)
    print(model, "\n", classification_report(y_holdout, y_holdout_pred), "\n\n")
