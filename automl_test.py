import pandas as pd
from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np

test_data = TabularDataset('test.csv')
predictor_new = TabularPredictor.load('/yshare2/ZETTAI_path_WA_slash_home_KARA/home/zeheng/DS/rp1/AutogluonModels/ag-20211222_015432')
label = 'shares'
preds = predictor_new.predict(test_data)
submission = pd.DataFrame({label:preds})
submission.to_csv('automl_full_wolog.txt', index=False)