from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.pretraining import TabNetPretrainer
import pandas as pd
import numpy as np
import random
import torch

random.seed(0)
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

unsupervised_model = TabNetPretrainer(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    mask_type="sparsemax"
)

unsupervised_model.fit(
    X_train=X_train,
    eval_set=[X_val],
    max_epochs=1000 ,
    patience=50,
    pretraining_ratio=0.5,
)

unsupervised_model.save_model('./test_h_wolog')
