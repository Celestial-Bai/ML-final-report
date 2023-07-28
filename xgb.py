import xgboost as xgb
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('train_new.csv')
val_data = pd.read_csv('val.csv')
X_train = train_data.iloc[:, 0:36]
X_train['n_tokens_title'] = -abs(X_train['n_tokens_title'] - 11.5)
X_train['var_shares'] = (X_train['self_reference_min_shares'] + X_train['self_reference_max_shares'] - 2*X_train['self_reference_avg_sharess'])
y_train = np.log(train_data.iloc[:, -1])
X_val = val_data.iloc[:, 0:36]
X_val['n_tokens_title'] = -abs(X_val['n_tokens_title'] - 11.5)
X_val['var_shares'] = (X_val['self_reference_min_shares'] + X_val['self_reference_max_shares'] - 2*X_val['self_reference_avg_sharess'])
y_val = np.log(val_data.iloc[:, -1])
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'max_depth': 5,
    'eta': 0.05,
    'eval_metric':'mae',
    'subsample': 0.7,
    'seed': 0
}
plst = params
evallist = [(dtrain, 'train'), (dval, 'eval')]
num_round = 13
bst = xgb.train(plst, dtrain, num_round, evallist, early_stopping_rounds=30)
test_data = pd.read_csv('test_new.csv')
test_data['n_tokens_title'] = -abs(test_data['n_tokens_title'] - 11.5)
test_data['var_shares'] = (test_data['self_reference_min_shares'] + test_data['self_reference_max_shares'] - 2*test_data['self_reference_avg_sharess'])
dtest = xgb.DMatrix(test_data)
pred = np.exp(bst.predict(dtest))

f = open('xgb_mh_wlog_1.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()
