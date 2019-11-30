import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

# download training data using kaggle api and load into pandas dataframe
os.system("kaggle competitions download -f train.csv house-prices-advanced-regression-techniques")
train = pd.read_csv("train.csv", index_col = 0)
os.system("rm train.csv")

# create a dataframe with only the training columns that are numeric
train_vet = pd.DataFrame(index = train.index)

for col in train.columns:
    if train[col].dtype != object:
        train_vet = pd.concat([train_vet, train[col]], 
                              axis = 1,
                             ignore_index = False)

train_vet = train_vet.fillna(0)

# fit linear model on numeric columns
lm = LinearRegression()

X = train_vet.iloc[:, 0:(len(train_vet.columns) - 1)]
y = train_vet['SalePrice']

lm.fit(X, y)

# download test dataset using kaggle api and load into pandas dataframe
os.system("kaggle competitions download -f test.csv house-prices-advanced-regression-techniques")
test = pd.read_csv("test.csv", index_col = 0)
os.system("rm test.csv")

# create test prediction input
test_vet = pd.DataFrame(index = test.index)

for col in test.columns:
    if test[col].dtype != object:
        test_vet = pd.concat([test_vet, test[col]],
                            axis = 1,
                            ignore_index = False)
        
test_vet = test_vet.fillna(0)

X2 = test_vet

# creating submission dataframe
submission = pd.DataFrame(index = test_vet.index)

submission = pd.concat([submission, pd.DataFrame(lm.predict(X2), index = test_vet.index).round(2)],
                      axis = 1,
                      ignore_index = False)

submission.columns = ["SalePrice"]
submission.index.name = "Id"

# submitting submission dataframe via kaggle api
submission.to_csv('submission.csv')
os.system("kaggle competitions submit house-prices-advanced-regression-techniques -f submission.csv -m 'simple linear model predictions'")
os.system("rm submission.csv")