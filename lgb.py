import lightgbm as lgb
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('train_full.csv')
val_data = pd.read_csv('val_full.csv')
X_train = train_data.iloc[:, 2:60]
X_train['n_tokens_title'] = -abs(X_train['n_tokens_title'] - 11.5)
X_train['var_shares'] = (X_train['self_reference_min_shares'] + X_train['self_reference_max_shares'] - 2*X_train['self_reference_avg_sharess'])
y_train = np.log(train_data.iloc[:, -1])
X_val = val_data.iloc[:, 2:60]
X_val['n_tokens_title'] = -abs(X_val['n_tokens_title'] - 11.5)
X_val['var_shares'] = (X_val['self_reference_min_shares'] + X_val['self_reference_max_shares'] - 2*X_val['self_reference_avg_sharess'])
y_val = np.log(val_data.iloc[:, -1])
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    'task': 'train',
    'boosting': 'gbdt',
    'objective': 'regression_l1',
    'num_leaves': 31,
    'metric': 'mae',
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    'boost_from_average': True
}
gbm = lgb.train(params,lgb_train,num_boost_round=800,valid_sets=lgb_val,early_stopping_rounds=30)
test_data = pd.read_csv('test.csv')
test_data['n_tokens_title'] = -abs(test_data['n_tokens_title'] - 11.5)
test_data['var_shares'] = (test_data['self_reference_min_shares'] + test_data['self_reference_max_shares'] - 2*test_data['self_reference_avg_sharess'])
pred = np.exp(gbm.predict(test_data))

f = open('lgb_full_wlog_1.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()