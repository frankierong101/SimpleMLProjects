from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, "titanic_data.csv")
df = pd.read_csv(file_path)

##Cleaning
common = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(common)
avg = df["Age"].mean()
df["Age"] = df["Age"].fillna(avg)
df = df.drop('Cabin', axis=1)
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

##Checks made when cleaning
#print(df["Embarked"].unique())
#print(df.isna().sum())
#print(df.columns.tolist())
#print(df.duplicated().sum())