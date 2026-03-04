import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
TRAINING_DATA_PATH = BASE_DIR / "Training" / "training_data_VT2026.csv"
TEST_DATA_PATH = BASE_DIR / "Test" / "test_data_VT2026.csv"


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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


def get_pipeline(model):
    feature_step = FunctionTransformer(feature_engineering, validate=False)

    # Vi väljer samma logik som i gamla:
    # - day_of_week one-hot
    # - skala alla numeriska utom snow_or_not och summertime
    # ColumnTransformer kan välja kolumner via "remainder" om vi gör två transformers:
    # 1) cat: day_of_week
    # 2) num: alla andra numeriska (men EXKLUDERA snow_or_not & summertime)
    #
    # För att kunna välja "alla utom" behöver vi lista kolumner -> gör det via en callable selector.

    def num_selector(X: pd.DataFrame):
        # Efter feature_engineering
        exclude = {"snow_or_not", "summertime"}
        # ta numeriska kolumner som inte är exkluderade och inte day_of_week
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

def load_training_data():
    X, y = load_full_training_data()
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y
    )
    return X_train, X_holdout, y_train, y_holdout

def load_full_training_data():
    df = pd.read_csv(TRAINING_DATA_PATH)
    X = df.drop("increase_stock", axis=1)
    y = df["increase_stock"]
    return X, y

def load_test_data():
    return pd.read_csv(TEST_DATA_PATH)