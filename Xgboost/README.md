# Xgboost model

Using data from a [Video games Sales](https://www.kaggle.com/datasets/gregorut/videogamesales) dataset, I extracted and made new features to more accurately predict my target which was global sales. I learnt regex to make a feature that checks whether a game was a sequel or not based on the title and outputs it to a binary feature. 

To further increase the R² score on my model, I tried:
 - Reducing the features dimensions with PCA
 - Scaling the numerical features with different types of scalers
 - Removing top % outliers and imputing instead of removal of rows 
   
However they didn't help with my R² score.

After feature engineering, I learnt the basics of hyperparameter tuning to use `RandomisedSearchCV()` for my model. I ran the model multiple times while adjusting the parameter ranges to maximise model accuracy. 

Finally, I used cross validation to choose the optimal number of steps/rounds to train my model with. This helped with preventing overfitting.

## Steps to run
1. Enter github codespaces for this codespace

2. Copy paste the two following commands one by one into the terminal
```sh
cd Xgboost
```
```sh
python Xgboost.py
```
You'll see the terminal log outputs every 100 rounds for the CV with the best round shown as well. Then the RMSE and R² score from the model is shown along with the total amount of time it took to run my script from start to finish.