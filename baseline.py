import pandas as pd
import random
import os
import numpy as np
import xgboost as xgb

from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_squared_error

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(42) # Seed 고정

train_df = pd.read_csv('./dataset/train.csv')

train_x = train_df.filter(regex='X') # Input : X Featrue
train_y = train_df.filter(regex='Y') # Output : Y Feature

#SVR = MultiOutputRegressor(SVR(), n_jobs=-1).fit(train_x, train_y) SVR을 이용한 Regression
#LSVR = MultiOutputRegressor(LinearSVR(), n_jobs=-1).fit(train_x, train_y)
#LR = MultiOutputRegressor(LinearRegression()).fit(train_x, train_y)
#XGB = MultiOutputRegressor(xgb.XGBRegressor(n_estimators=150, learning_rate=0.07, gamma = 0, subsample=0.77, colsample_bytree = 1, max_depth=7) ).fit(train_x, train_y) 
XGB = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear', n_estimators=50, learning_rate=0.1, alpha = 10, gamma = 0, subsample=0.77, colsample_bytree = 0.3, max_depth=5) ).fit(train_x, train_y) 


print('Done.')

test_x = pd.read_csv('./dataset/test.csv').drop(columns=['ID'])

preds = XGB.predict(test_x)
print('Done.')

print('calculate rmse...')

rmse = np.sqrt(mean_squared_error(test_x, preds), squared=False)
print("RMSE: %f" (rmse))

submit = pd.read_csv('./dataset/sample_submission.csv')

for idx, col in enumerate(submit.columns):
    if col=='ID':
        continue
    submit[col] = preds[:,idx-1]
print('Done.')

submit.to_csv('./dataset/submit_xgb.csv', index=False)
