from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
import pandas as pd
import numpy as np
import random

random.seed(0)
loaded_pretrain = TabNetPretrainer()
loaded_pretrain.load_model('./test_pretrain.zip')
train_data = pd.read_csv('tr_new.csv')
val_data = pd.read_csv('val.csv')
X_train = train_data.iloc[:, 0:36]
X_train = np.array(X_train, dtype = np.float32)
y_train = np.log(train_data.iloc[:, -1])
y_train = y_train[:, np.newaxis]
X_val = val_data.iloc[:, 0:36]
X_val = np.array(X_val, dtype = np.float32)
y_val = np.log(val_data.iloc[:, -1])
y_val = y_val[:, np.newaxis]

clf = TabNetRegressor()
clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_name=['train', 'valid'],
    eval_metric=['mae'],
    max_epochs=300,
    patience=30,
    num_workers=0,
    drop_last=False,
    from_unsupervised=loaded_pretrain
)

test_data = pd.read_csv('test_new.csv')
test_data = np.array(test_data, dtype = np.float32)
pred = np.exp(clf.predict(test_data))

f = open('tabnet_mh_wlog1.txt',mode='w')
for i in range(len(pred)):
    print(pred[i][0], file=f)
f.close()