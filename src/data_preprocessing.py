import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
from sklearn.preprocessing import StandardScaler

training_data_VT2026 = pd.read_csv('../Training/training_data_VT2026.csv')

# In this file df is altered. Then df_scaled is made with one command.

#copy data
df = training_data_VT2026.copy()

#make months and hour of days make sense by making in circular
N1 = 12
df['month_sin'] = np.sin(2 * np.pi * df['month'] / N1)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / N1)

N3 = 24
df['hour_of_the_day_sin'] = np.sin(2* np.pi * df['hour_of_day'] / N3)
df['hour_of_the_day_cos'] = np.cos(2* np.pi * df['hour_of_day'] / N3)

# one hot encode day of week 
df = pd.get_dummies(df, columns=['day_of_week'], prefix='day')

#make snow binary
df['snow_or_not'] = (df['snowdepth'] > 0).astype(int)

#drop original month, hour of day, snowdepth and snow columns
df = df.drop('month', axis=1)
df = df.drop('hour_of_day', axis=1)
df = df.drop('snowdepth', axis=1)
df = df.drop('snow', axis=1)

#split into features and target variable
X = df.drop('increase_stock', axis=1)
y = df['increase_stock']

#scaled version od data. standard scaler is used i.e. x = (x - mean) / std. we avoid to scale days 

# identify day columns
day_cols = ['day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6']
cols_to_scale = X.columns.difference(day_cols)
X_scaled_part = StandardScaler().fit_transform(X[cols_to_scale])
# Gör tillbaka till DataFrame
X_scaled_part = pd.DataFrame(
    X_scaled_part,
    columns=cols_to_scale,
    index=X.index
)
# Lägg tillbaka day_cols utan scaling
X_scaled = pd.concat([X_scaled_part, X[day_cols]], axis=1)

# (valfritt) behåll ursprunglig kolumnordning
X_scaled = X_scaled[X.columns]


#combine scaled features with target variable. Ready to use.
df_scaled = pd.concat([X_scaled, y], axis=1)
print(X_scaled.head())


