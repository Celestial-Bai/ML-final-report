import pandas as pd
import numpy as np
import autogluon.text
from autogluon.tabular import TabularDataset, TabularPredictor

train_data = TabularDataset('tr_whole.csv')
label = 'shares'
train_data['shares'] = np.log(train_data['shares'])
predictor = TabularPredictor(label=label, problem_type = 'regression', eval_metric = 'mean_absolute_error').fit(train_data, excluded_model_types = ['NN'])
