import os
import pandas as pd
from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

## Initialise
model = LogisticRegression(max_iter=1000)


script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'titanic_data.csv')
df = pd.read_csv(file_path)

## Cleaning
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
# df['FamilySize'] = df['Sibsp'] + df['Parch'] accuracy -= 0.01 when used

## Checks made when cleaning
# print(df["Embarked"].unique())
# print(df.isna().sum())
# print(df.columns.tolist())
# print(df.duplicated().sum())

## Features & Target
x = df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']]
y = df['Survived']

## Splitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2004)

## Standardisation doesn't affect the accuracy at all
# scaler = StandardScaler()
# x_train_scaled = scaler.fit_transform(x_train)
# x_test_scaled = scaler.transform(x_test)

## Fitting
model.fit(x_train, y_train)

## Accuracy Check
y_prediction = model.predict(x_test)
accuracy = accuracy_score(y_test, y_prediction) * 100
print(f"Accuracy: {accuracy:.2f}%")