import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

## Initialise
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'vgsales.csv')
df = pd.read_csv(file_path)

## Cleaning + Encoding
# print(df.isnull().sum())
df = df.dropna(subset=(df.columns.tolist()))
# print(df.isnull().sum())

# print(df.columns.tolist())

x_categoricals = encoder.fit_transform(df[['Platform', 'Genre', 'Publisher']])
x_num = df[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].to_numpy()
x = np.hstack([x_num, x_categoricals])
y = df['Global_Sales']

## Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2004)

## Training
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

## Evalutating

y_prediction = rf.predict(x_test)
# print(y_prediction)
r2 = r2_score(y_test, y_prediction)
rmse = root_mean_squared_error(y_test, y_prediction)
print(f"Test RMSE: {rmse:.3f}, R²: {r2:.3f}")