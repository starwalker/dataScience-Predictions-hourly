from TextMiningMachine.io import get_data
from TextMiningMachine.feature_extraction import DataSetBuilder
from TextMiningMachine.xgboost_optimizer import XgboostOptimizer
import pandas as pd
import pickle
import xgboost as xgb
from nltk.corpus import stopwords
import numpy as np
from datetime import timedelta
import datetime
import os
import gc
from scipy import sparse

data = pd.read_pickle('data/raw_data.p')
print(data.shape[0])
trans = pickle.load(open('dsvm_build_files/text_cat_transformer.p','rb'))
date_col = 'REPORTING_TIME'


features = sparse.load_npz('data/features.npz')
features = sparse.csr_matrix(features)
print(features.shape)
keep_cols = ['PAT_ID', 'PAT_ENC_CSN_ID', 'REPORTING_TIME']
#target_cols = ['MetSIRS4_4hr_8', 'MetSIRS4_4hr_24', 'MetSIRS4_4hr_48']
target_col = 'MetSIRS4_4hr_24'



#
#
# keep_inds =np.where((data['NoSIRS4_4hr_4']==1) | (data['MetSIRS4_4hr']==1))[0]
# data = data.iloc[keep_inds,:].reset_index(drop = True)
# features = features[keep_inds,:]

weights = np.ones((data.shape[0],1))
weights[np.where((data['SIRS4_4hr_Countdown']<=24) & (data['SIRS4_4hr_Countdown']>0))[0]] = 6

#split the date into 70%, 20%, 10% for train, test, eval resp.
train_cut_date = data[date_col].sort_values().reset_index(drop=True)[int(data.shape[0] *.60)]
test_cut_date = data[date_col].sort_values().reset_index(drop=True)[int(data.shape[0] *.80)]

train_inds = np.where(data[date_col]<=train_cut_date)[0]
test_inds = np.where((data[date_col]>train_cut_date) & (data[date_col]<=test_cut_date))[0]
eval_inds = np.where(data[date_col]>test_cut_date)[0]


train_x = sparse.coo_matrix(features[train_inds,:])
test_x = sparse.coo_matrix(features[test_inds,:])
eval_x = sparse.coo_matrix(features[eval_inds,:])

gc.collect()
col_ind = list(data.columns).index(target_col)
train = xgb.DMatrix(train_x, feature_names=trans.feature_names,label=data.iloc[train_inds,col_ind].values, weight= weights[train_inds])
test = xgb.DMatrix(test_x, feature_names=trans.feature_names,label=data.iloc[test_inds,col_ind].values, weight= weights[test_inds])
eval = xgb.DMatrix(eval_x, feature_names=trans.feature_names,label=data.iloc[eval_inds,col_ind].values, weight= weights[eval_inds])


#w Build XGboost models
x = XgboostOptimizer()
# update features related to random search algorithm
x.update_params({'objective': 'binary:logistic',
                 'eval_metric':'auc',
                 'max_over_fit': .03,
                 'num_rand_samples' : 40,
                 'eta': np.arange(.04, .25, .01),
                 'max_depth': range(2, 9),
                 'min_child_weight': range(3000, 20000, 300),
                 'num_boost_rounds': 100,
                 'early_stopping_rounds': 2,
                 'max_minutes_total':60})

#call random search algorithm
x.fit_random_search(dtrain = train, evals=[(train, 'Train'), (test, "Test")])


# extract the best model
model = x.best_model
model.feature_names = trans.feature_names


file = 'models/xgb/'+target_col+'Uniform6_Model.p'
pickle.dump(model, open(file, 'wb'))