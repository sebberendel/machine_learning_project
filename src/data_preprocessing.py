import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ----------------------------------------------------
# Paths to data
# ----------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = BASE_DIR / "Training" / "training_data_VT2026.csv"
TEST_DATA_PATH = BASE_DIR / "Test" / "test_data_VT2026.csv"

# ----------------------------------------------------
# Feature engineering function
# ----------------------------------------------------
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformations to raw dataframe:
    - Circular encoding for month & hour_of_day
    - Binary snow indicator
    - Drop replaced/unnecessary columns
    """
    df = df.copy()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["hour_of_day_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_of_day_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    df["snow_or_not"] = (df["snowdepth"] > 0).astype(int)

    df = df.drop(
        columns=["month", "hour_of_day", "snowdepth", "snow", "holiday", "weekday"],
        errors="ignore"
    )
    return df

# ----------------------------------------------------
# Pipeline builder
# ----------------------------------------------------
def get_pipeline(model):
    """
    Build a pipeline that:
    1. Applies feature engineering
    2. One-hot encodes 'day_of_week'
    3. Scales numeric columns except 'snow_or_not' & 'summertime'
    4. Fits the provided model
    """
    feature_step = FunctionTransformer(feature_engineering, validate=False)

    def num_selector(X: pd.DataFrame):
        # Select numeric columns to scale, excluding binary columns
        exclude = {"snow_or_not", "summertime"}
        cols = X.select_dtypes(include=[np.number]).columns
        cols = [c for c in cols if c not in exclude]
        return cols

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["day_of_week"]),
            ("scale", StandardScaler(), num_selector),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    return Pipeline([
        ("feature_engineering", feature_step),
        ("preprocess", preprocessor),
        ("model", model),
    ])

# ----------------------------------------------------
# Training data loader (splits)
# ----------------------------------------------------
def load_training_data():
    """
    Load training data and split into training and holdout sets.
    Returns: X_train, X_holdout, y_train, y_holdout
    """
    X, y = load_full_training_data()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    return X_train, X_holdout, y_train, y_holdout

# ----------------------------------------------------
# Full training data loader
# ----------------------------------------------------
def load_full_training_data():
    """
    Load the full training dataset without splitting.
    Returns: X, y
    """
    df = pd.read_csv(TRAINING_DATA_PATH)
    X = df.drop("increase_stock", axis=1)
    y = df["increase_stock"]
    return X, y

# ----------------------------------------------------
# Test data loader
# ----------------------------------------------------
def load_test_data():
    """Load the test dataset."""
    return pd.read_csv(TEST_DATA_PATH)