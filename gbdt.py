from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('train_full.csv')
val_data = pd.read_csv('val_full.csv')
X_train = train_data.iloc[:, 2:60]
X_train['n_tokens_title'] = -abs(X_train['n_tokens_title'] - 11.5)
X_train['var_shares'] = (X_train['self_reference_min_shares'] + X_train['self_reference_max_shares'] - 2*X_train['self_reference_avg_sharess'])
#X_train['self_reference_avg_sharess'] = X_train['self_reference_avg_sharess']**2
y_train = np.log(train_data.iloc[:, -1])
X_val = val_data.iloc[:, 2:60]
X_val['n_tokens_title'] = -abs(X_val['n_tokens_title'] - 11.5)
X_val['var_shares'] = (X_val['self_reference_min_shares'] + X_val['self_reference_max_shares'] - 2*X_val['self_reference_avg_sharess'])
#X_val['self_reference_avg_sharess'] = X_val['self_reference_avg_sharess']**2
y_val = np.log(val_data.iloc[:, -1])
params = {'n_estimators': 800,
          'max_depth': 5,
          'learning_rate': 0.05,
           'loss': 'lad'}
model = GradientBoostingRegressor(**params)
model.fit(X_train,y_train)
test_data = pd.read_csv('test.csv')
test_data['n_tokens_title'] = -abs(test_data['n_tokens_title'] - 11.5)
test_data['var_shares'] = (test_data['self_reference_min_shares'] + test_data['self_reference_max_shares'] - 2*test_data['self_reference_avg_sharess'])
#test_data['self_reference_avg_sharess'] = test_data['self_reference_avg_sharess']**2
pred = np.exp(model.predict(test_data))

f = open('gbdt_full_wlog_att4.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()