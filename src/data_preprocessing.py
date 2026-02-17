import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "Training" / "training_data_VT2026.csv"

training_data_VT2026 = pd.read_csv(DATA_PATH)


# Copy data
df = training_data_VT2026.copy()

# --- Circular encoding ---
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['hour_of_day_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_of_day_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

# --- One-hot encode day of week ---
df = pd.get_dummies(df, columns=['day_of_week'], prefix='day')

# --- Binary snow variable ---
df['snow_or_not'] = (df['snowdepth'] > 0).astype(int)

# --- Drop unused / replaced columns ---
df = df.drop(
    columns=[
        'month',
        'hour_of_day',
        'snowdepth',
        'snow',
        'holiday',   # droppas enligt krav
        'weekday'    # droppas enligt krav
    ]
)

# --- Split features / target ---
X = df.drop('increase_stock', axis=1)
y = df['increase_stock']

# --- Columns that should NOT be scaled ---
day_cols = [c for c in X.columns if c.startswith("day_")]
binary_cols = ['snow_or_not', 'summertime']

no_scale_cols = day_cols + binary_cols

# --- Columns to scale ---
cols_to_scale = X.columns.difference(no_scale_cols)

# --- Scale ---
scaler = StandardScaler()
X_scaled_part = scaler.fit_transform(X[cols_to_scale])

X_scaled_part = pd.DataFrame(
    X_scaled_part,
    columns=cols_to_scale,
    index=X.index
)

# --- Combine scaled + unscaled ---
X_scaled = pd.concat(
    [X_scaled_part, X[no_scale_cols]],
    axis=1
)

# --- Restore original column order ---
X_scaled = X_scaled[X.columns]

# --- Final dataset ---
df_scaled = pd.concat([X_scaled, y], axis=1)

print(X_scaled.describe())

# - Pipeline ----------------------------------------------
def get_pipeline(model):
    # cols_to_scale = [col for col in X.columns if col not in ['increase_stock'] and col not in ['day_Monday', 'day_Tuesday', 'day_Wednesday', 'day_Thursday', 'day_Friday', 'day_Saturday', 'day_Sunday', 'snow_or_not', 'summertime']]
    preprocessor = ColumnTransformer(
        transformers=[
            ("scale", StandardScaler(), cols_to_scale)
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model)
    ])
    return pipeline