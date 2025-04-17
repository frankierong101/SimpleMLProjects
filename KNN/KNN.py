import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

## Initialise
le = LabelEncoder()
scaler = MinMaxScaler()
# TRY THIS AND COMPARE 
# scaler = StandardScaler()
knn = NearestNeighbors(n_neighbors=6, metric='cosine')

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'vgsales.csv')
df = pd.read_csv(file_path)

## Cleaning 
# print(f" Before: {len(df)}")
df = df.dropna(subset=['Name', 'Platform', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year'])
# print(f" After: {len(df)}")

# TRY THIS AND COMPARE 
# df = df.dropna(subset=(df.columns.tolist()))


df['Total_Sales'] = df['NA_Sales'] + df['EU_Sales'] + df['JP_Sales'] + df['Other_Sales']
# print(df['Total Sales'])
# print(df.columns.tolist())
df_recommend = df[['Name', 'Platform', 'Genre', 'Publisher', 'Total_Sales', 'Year']].copy()

df_recommend['Year'] = scaler.fit_transform(df_recommend[['Year']])
df_recommend['Platform'] = le.fit_transform(df_recommend['Platform'])
df_recommend['Genre'] = le.fit_transform(df_recommend['Genre'])
df_recommend['Publisher'] = le.fit_transform(df_recommend['Publisher'])
# print(df_recommend['Platform']) 
# print(df['Platform'].unique())

## Fitting
knn.fit(df_recommend[['Platform', 'Genre', 'Publisher', 'Total_Sales', 'Year']])

def recommend(game_name):
    #Finding game index
    game_index = df_recommend[df_recommend['Name'] == game_name].index[0]
    
    #Finding similar games
    features_df = df_recommend.loc[[game_index], ['Platform', 'Genre', 'Publisher', 'Total_Sales', 'Year']]
    distances, indices = knn.kneighbors(features_df)

    print(f"Recommendations for {game_name}:")
    print("-"*30)
    for i in indices[0]:
        if i != game_index:
            print(df_recommend.iloc[i]['Name'])
    print("-"*30)

named_game = input(str("Insert game: "))
recommend(named_game)