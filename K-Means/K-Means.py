import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

## Initialising
scaler = StandardScaler()

script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, 'AC_catalog.csv')
df = pd.read_csv(file_path)

## Converting + Cleaning
# Str -> Int
df['Buy'] = pd.to_numeric(df['Buy'], errors='coerce')
df['Sell'] = pd.to_numeric(df['Sell'], errors='coerce')

# Drop empties and < 0, then drop extreme outliers
df = df.dropna(subset=['Buy', 'Sell'])
df = df[(df['Sell'] >= 0) & (df['Buy'] >= 0)]

df = df[(df['Buy'] < df['Buy'].quantile(0.98)) & (df['Sell'] < df['Sell'].quantile(0.98))]

# Third Feature
df['Price_Ratio'] = df['Buy'] / df['Sell']

features = ['Buy', 'Sell', 'Price_Ratio']
x = df[features]

## Scaling
x_scaled = scaler.fit_transform(x)

## Fitting + Running
k = 4

kmeans = KMeans(n_clusters=k, n_init=10, random_state=2004)
clusters = kmeans.fit_predict(x_scaled)
df['Cluster'] = clusters

## Metrics + Stats
silhouette_avg = silhouette_score(x_scaled, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.3f}")
print("+" + "-" * 22 + "+")
cluster_counts = df['Cluster'].value_counts()
print(cluster_counts)
print("+" + "-" * 43 + "+")
cluster_stats = df.groupby('Cluster')[['Buy', 'Sell', 'Price_Ratio']].mean()
print(cluster_stats)