import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = BASE_DIR / "Training" / "training_data_VT2026.csv"
TEST_DATA_PATH = BASE_DIR / "Test" / "test_data_VT2026.csv"

# ==========================================
# Feature engineering (GLOBAL – pickle safe)
# ==========================================

def feature_engineering(df):
    df = df.copy()

    # --- Circular encoding ---
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["hour_of_day_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_of_day_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)

    # --- Binary snow variable ---
    df["snow_or_not"] = (df["snowdepth"] > 0).astype(int)

    # --- Drop replaced columns ---
    df = df.drop(
        columns=[
            "month",
            "hour_of_day",
            "snowdepth",
            "snow",
            "holiday",
            "weekday",
        ],
        errors="ignore"  # extra safety
    )

    return df


# ==========================================
# Pipeline builder (clean & stable)
# ==========================================

def get_pipeline(model):

    feature_step = FunctionTransformer(feature_engineering)

    preprocessor = ColumnTransformer(
        transformers=[
            # One-hot encode only categorical column
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["day_of_week"])
        ],
        remainder="passthrough"  # keep all other columns as-is
    )

    pipeline = Pipeline(
        steps=[
            ("feature_engineering", feature_step),
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


# ==========================================
# Training data loader
# ==========================================

def load_training_data():

    df = pd.read_csv(TRAINING_DATA_PATH)

    X = df.drop("increase_stock", axis=1)
    y = df["increase_stock"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y  # viktigt vid obalanserade klasser
    )

    return X_train, X_holdout, y_train, y_holdout


# ==========================================
# Test data loader
# ==========================================

def load_test_data():

    df = pd.read_csv(TEST_DATA_PATH)

    return df