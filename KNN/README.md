# KNN Game Recommender
I used the [Video games Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) dataset from Kaggle and built a simple game recommender using KNN. I had a lot of features for each game from this dataset so KNN could effectively recommend games based on feature distance.

There aren't numerical metrics that I could measure my KNN model with since it's unsupervised. I tried using a `standard scaler()` instead of the `minmaxscaler()` but both netted me the same results.

## Steps to run
1. Enter github codespaces for this repository 

2. Copy paste the two following commands one by one into the terminal
```sh
cd KNN
```
```sh
python KNN.py
```
3. You'll be prompted and you can enter a game. You can try some of these examples below
- Just Dance
- Mario Kart Wii
- Tetris 
- Grand Theft Auto V
- Halo 3