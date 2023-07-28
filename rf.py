from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random

random.seed(0)
train_data = pd.read_csv('tr_high.csv')
X_train = train_data.iloc[:, 0:17]
y_train = np.log(train_data.iloc[:, -1])
forest = RandomForestRegressor(
    n_estimators=1000,
    random_state=0,
    criterion='mae',
    n_jobs=-1)
forest.fit(X_train,y_train)
test_data = pd.read_csv('test_high.csv')
pred = np.exp(forest.predict(test_data))

f = open('rf_h_wlog.txt',mode='w')
for i in range(len(pred)):
    print(pred[i], file=f)
f.close()