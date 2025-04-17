import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

## Initialise
model = LinearRegression()

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "house_data.csv")
df = pd.read_csv(file_path)

# Features and target
X = df[["SquareFeet"]]
y = df["Price"]

# Fitting
model.fit(X, y)

# SoloPrediction
X_new = pd.DataFrame([[1000]], columns=["SquareFeet"])
y_pred = model.predict(X_new)
print(f"Predicted value: {y_pred[0]:.2f}")

# EvaluationMetrics
mse = mean_squared_error(y, model.predict(X))
print(f"MSE: {mse:.2f}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")