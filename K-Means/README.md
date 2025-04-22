# Kmeans Model

I used an [Animal Crossing dataset](https://www.kaggle.com/datasets/jessicali9530/animal-crossing-new-horizons-nookplaza-dataset) as the source to run a Kmeans model. In my case it clustered 4 different groups of items together.

At first I selected the most meaningful features which cut 22 columns down to 10 columns; 2 were integers, 2 were booleans and 6 was string format. It ran really weirdly with a terrible silhouette score and was saying 10 clusters was optimal, after some trial and error I found out removing all boolean features and then string features improved the model. I ultimately ended up only using the two integer features and engineered a third feature (line 26) based on the relationship of those two integer features. 

## Steps to run
1. Enter github codespaces for this codespace

2. Copy paste the two following commands one by one into the terminal
```sh
cd K-Means
```
```sh
python K-Means.py
```
You'll see the silhoutte score, cluster amount + size and each clusters information