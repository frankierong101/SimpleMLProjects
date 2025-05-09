import os, re
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

## Loading data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'vgsales.csv')).dropna()
start_time = time.time()

## Removing outliers
# print(len(df['Global_Sales']))
# low, high = df['Global_Sales'].quantile([0.05, 0.95])
# df = df[(df['Global_Sales'] > low) & (df['Global_Sales'] < high)]
# print(len(df['Global_Sales']))

## Feature engineering
pattern = r'''
    \b(?:[IVX]+|\d+)\b                # Roman numerals (VII, X) or standalone digits (2, 3)
    |                                  
    (?:Part|Episode|Volume|Expansion|DLC|Season)\s+\d+  # "Part 2", "DLC 3"
    |                                  
    \d+(?:st|nd|rd|th)\b              # Ordinals like "1st", "2nd" (less common but possible)
    |                                  
    \b\d+[-:]\d+\b                    # Hyphenated/colon formats (e.g., "X-2", "2-3")
    |                                  
    \b\d+[kK]\d*\b                    # Alphanumeric patterns like "2K24", "3K"
'''
regex = re.compile(pattern, flags=re.IGNORECASE | re.VERBOSE)
df['Sequel'] = df['Name'].str.contains(regex, na=False).astype(int)

df['TitleLength'] = df['Name'].str.len()
df['YearsSinceRelease'] = 2025 - df['Year']
df['Decade'] = (df['Year'] // 10) * 10
df['Platform_Age'] = 2025 - df['Platform'].map({'Wii': 2006, 'PS4': 2013, 'X360': 2005, 'PS3': 2006, 'PS2': 2000, 'XOne': 2013,'GBA': 2001, 'DS': 2004, 'PS': 1994, 'SNES': 1990, 'GEN': 1988, 'N64': 1996, '2600': 1977, 'PSP': 2004, 'PSV': 2011, '3DS': 2011, 'SAT': 1994, 'DC': 1999})
df['Platform_Age_Squared'] = df['Platform_Age'] ** 2
df['TitleWordCount'] = df['Name'].str.split().str.len()
df['DominantRegion'] = df[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].idxmax(axis=1)

encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
# scaler = StandardScaler()

# Combining and splitting data
x_categorical = encoder.fit_transform(df[['Platform', 'Genre', 'Publisher', 'DominantRegion']])
x_numerical = df[['Year','TitleLength','YearsSinceRelease','Decade','Platform_Age','Platform_Age_Squared', 'TitleWordCount', 'Sequel']].values
# x_numerical = scaler.fit_transform(x_numerical)
x_features = np.hstack([x_numerical, x_categorical])

# y_target = df['Global_Sales']
y_target = np.log1p(df['Global_Sales'].values)
y_target = np.sqrt(y_target)


x_train, x_test, y_train, y_test = train_test_split(x_features, y_target, test_size=0.2, random_state=1000)

dtrain = xgb.DMatrix(x_train, label=y_train) # Fitting into a matrix to run xgboost with GPU support
###################################################################################### HYPERPARAMETER TUNING
#  random_param_grid = {
#     'subsample': [0.4, 0.5, 0.6],
#     'max_depth': [8, 9, 10],
#     'learning_rate': [0.01, 0.02, 0.03],
#     'colsample_bytree': [0.7, 0.8, 0.9],
#     # 'n_estimators': [500, 600, 700]
# }

# xgb_model = xgb.XGBRegressor(
#     objective='reg:squarederror',
#     tree_method='hist',  # Faster historgram tree method
#     device='cuda',  # Train on GPU
#     random_state=2004
# )

# random_search = RandomizedSearchCV(
#     estimator=xgb_model,
#     param_distributions=random_param_grid,
#     n_iter=1000,  # number of combinations to try
#     cv=10,       # cross-validation folds
#     verbose=2,
#     random_state=2004,
#     scoring='r2',
#     n_jobs=1  # uses 1 CPU core
# )

# # Use NumPy arrays for RandomizedSearchCV
# random_search.fit(x_train, y_train)
# print("Best parameters found: ", random_search.best_params_)
# print("Best r2 found: ", random_search.best_score_)
# best_params = random_search.best_params_
# print(best_params)
# params = {
#     'objective': 'reg:squarederror',
#     'tree_method': 'hist',
#     'device': 'cuda', 
#     'verbosity': 2,
#     'random_state': 2004,
#     # 'seed': 2004,
#     **best_params
# }
######################################################################################
params = {
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'device': 'cuda',
    # 'booster': 'dart', 
    'verbosity': 1,
    'random_state': 2004,

    'max_depth': 9,
    'learning_rate': 0.02,
    'subsample': 0.5,
    'colsample_bytree': 0.8,
}


# Cross-validate with xgboost.cv on GPU
evals = xgb.cv(
    params,
    dtrain,
    num_boost_round=1201,
    nfold=15,
    metrics='rmse',
    early_stopping_rounds=20,
    seed=2004,
    as_pandas=True,
    verbose_eval=100
)

# Best number of rounds
best_rounds = len(evals)
print(f"Best rounds {best_rounds}")
# print(evals)
# print(evals.tail(1))

# Train final model with best rounds
model = xgb.train(params, dtrain, num_boost_round=best_rounds)

# Predict on test
dtest = xgb.DMatrix(x_test)
y_pred = model.predict(dtest)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Test RMSE: {rmse:.3f}")
print(f"Test R²: {r2:.3f}")

end_time = time.time()
print(f"Total runtime: {end_time - start_time:.2f} seconds")