import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, cross_val_predict, GridSearchCV
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, make_scorer, recall_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from data_preprocessing import load_training_data, get_pipeline


recall_high = make_scorer(
    recall_score,
    pos_label="high_bike_demand"
)

# ----------------------------------------------------
# Helpers: get preprocessed feature matrix from pipeline
# ----------------------------------------------------
def get_preprocessed_X(pipeline, X, y=None):
    """
    Fits and transforms X through the pipeline up to preprocess step,
    returning a dense numpy array for covariance/PCA/QDA plotting.
    """
    fe = pipeline.named_steps["feature_engineering"]
    pre = pipeline.named_steps["preprocess"]

    X_fe = fe.fit_transform(X, y)
    X_pre = pre.fit_transform(X_fe, y)

    if hasattr(X_pre, "toarray"):
        X_pre = X_pre.toarray()

    return X_pre


# ----------------------------------------------------
# 1) Frobenius norm between class covariances (on preprocessed X)
# ----------------------------------------------------
def frobenius_cov_diff(X, y, base_pipeline):
    X_pre = get_preprocessed_X(base_pipeline, X, y)

    labels = pd.Series(y).unique()
    if len(labels) != 2:
        raise ValueError(f"Expected binary classification, got labels: {labels}")

    y_ser = pd.Series(y).reset_index(drop=True)
    X_df = pd.DataFrame(X_pre)

    cov1 = X_df[y_ser == labels[0]].cov()
    cov2 = X_df[y_ser == labels[1]].cov()

    fro_norm = np.linalg.norm((cov1 - cov2).values, ord="fro")
    print(f"Frobenius norm of covariance diff between {labels[0]} and {labels[1]}: {fro_norm:.2f}")
    return fro_norm


# ----------------------------------------------------
# 2) Grid search for QDA (inside pipeline)
# ----------------------------------------------------
def qda_grid_search(X, y, priors=None, random_state=1):
    if priors is None:
        priors = [0.61, 0.39]

    qda = QuadraticDiscriminantAnalysis(priors=priors)
    pipe = get_pipeline(qda)

    param_grid = {"model__reg_param": np.linspace(0, 1, 20)}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring=recall_high,
        n_jobs=-1,
        refit=True
    )

    grid.fit(X, y)
    print("\n=== GRID SEARCH (QDA) ===")
    print("Best params:", grid.best_params_)
    print("Best CV score (accuracy):", grid.best_score_)
    return grid


# ----------------------------------------------------
# 3) CV classification report for best QDA params (fair comparison)
# ----------------------------------------------------
def qda_best_cv_report(X, y, best_reg_param, priors=None, random_state=1):
    if priors is None:
        priors = [0.61, 0.39]

    best_qda = QuadraticDiscriminantAnalysis(reg_param=best_reg_param, priors=priors)
    best_pipe = get_pipeline(best_qda)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    y_pred = cross_val_predict(best_pipe, X, y, cv=cv, n_jobs=-1)

    print("\n=== QDA (BEST PARAMS) - CV Classification Report ===")
    print("Best reg_param:", best_reg_param)
    print(classification_report(y, y_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(y, y_pred)
    plt.title("Confusion Matrix - QDA (Best Params, CV)")
    plt.show()

    return y_pred


# ----------------------------------------------------
# 4) Plot decision boundary in PCA space (preprocessed -> PCA -> QDA)
# ----------------------------------------------------
def plot_qda_decision_boundary_pca(X, y, reg_param, priors=None, random_state=1):
    if priors is None:
        priors = [0.61, 0.39]

    # Preprocess X (feature engineering + scaling + dummies)
    pipe_for_transform = get_pipeline(QuadraticDiscriminantAnalysis())
    X_pre = get_preprocessed_X(pipe_for_transform, X, y)

    # PCA to 2D
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_pre)

    # Encode labels for coloring
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y))

    # Fit QDA in PCA space
    qda = QuadraticDiscriminantAnalysis(reg_param=reg_param, priors=priors)
    qda.fit(X_pca, y)

    # Create mesh
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = qda.predict_proba(grid)[:, 1].reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.contour(xx, yy, Z, levels=[0.5], colors="red", linewidths=2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_enc, edgecolor="k")
    plt.title("QDA Decision Boundary (PCA on preprocessed features)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()


# ====================================================
# RUN
# ====================================================
X_train, X_holdout, y_train, y_holdout = load_training_data()

# 1) Frobenius norm
base_pipe = get_pipeline(QuadraticDiscriminantAnalysis())
frobenius_cov_diff(X_train, y_train, base_pipe)

# 2) Grid search (pipeline)
grid = qda_grid_search(X_train, y_train)

best_reg = grid.best_params_["model__reg_param"]

# 3) CV report for best params (fair comparison)
qda_best_cv_report(X_train, y_train, best_reg_param=best_reg)

# 4) Plot decision boundary in PCA space using best reg_param
plot_qda_decision_boundary_pca(X_train, y_train, reg_param=best_reg)


pipe = get_pipeline(QuadraticDiscriminantAnalysis(reg_param=0.0))
Xt = pipe.named_steps["feature_engineering"].fit_transform(X_train, y_train)
Xp = pipe.named_steps["preprocess"].fit_transform(Xt, y_train)
print("n_features_new:", Xp.shape[1])