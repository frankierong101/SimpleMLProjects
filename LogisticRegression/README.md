# Simple Logistic Regression Model
This is another simple model using logistic regression on the [titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) from Kaggle . I cleaned the data (lines 13–16) and trained the model, achieving 80% accuracy.

To increase the accuracy I tried combining two features into one (line 17) then ran it and lost 1% accuracy somehow. I also tried scaling the features (line 33-35) but I ended up with 80%.

## Steps to run
1. Enter github codespaces for this codespace

2. Copy paste the two following commands one by one into the terminal
```sh
cd LogisticRegression
```
```sh
python LogisticRegression.py
```
You'll see the outputted accuracy rate there