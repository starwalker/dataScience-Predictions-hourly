import pickle
import xgboost as xgb
import gc
import pandas as pd
import numpy as np
from TextMiningMachine.splitting import TrainTestDateSplitter
from TextMiningMachine.xgboost_optimizer import XgboostOptimizer
from TextMiningMachine.utils import generate_column_from_search
import xgboost as xgb
import math
import datetime
import sklearn
gc.collect()
import os


data_dir = 'data/all_raw_data/'
feature_dir = 'data/all_features/'
data_files = os.listdir(data_dir)
feature_files = os.listdir(feature_dir)
target_col = 'MetSIRS4_4hr_48'
date_col = 'REPORTING_TIME'


# def build_iterative_model(data_dir,feature_dir,num_passes_through_data,num_overall_trees,target_col,date_col):
#     data_files = os.listdir(data_dir)
#     feature_files = os.listdir(feature_dir)

if len(data_files)!=len(feature_files):
    print('There are not an equal number of data files as there are feature files, returning')
    pass
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
num_passes_through_data = 1
num_total_trees = 100
num_eval_files = 1
train_test_prop = .75
num_files = len(data_files)
eval_file_inds = list(range(num_files))[-num_eval_files:]
train_file_inds= list(range(num_files))[:(num_files-num_eval_files)]

# update features related to random search algorithm
params = {'objective': 'binary:logistic',
                 'eval_metric': ['logloss','auc' ],
                 'max_over_fit': .03,
                 'num_rand_samples': 50,
                 'num_boost_rounds': 100,
                 'eta': np.arange(.01, .25, .01),
                 'max_depth': range(1, 10),
                 'min_child_weight': range(5000, 25000, 1000),
                 'early_stopping_rounds': 25,
                 'build_past' : 3
                 }



if len(eval_file_inds)==1:
    eval_data = pd.read_pickle(data_dir+data_files[eval_file_inds[0]])
    eval_features = pd.read_pickle(feature_dir + feature_files[eval_file_inds[0]])
else:
    for i in range(len(eval_file_inds)):
        if i==0:
            eval_data = pd.read_pickle(data_dir + data_files[eval_file_inds[i]])
            eval_features = pd.read_pickle(feature_dir + feature_files[eval_file_inds[i]])
        else:
            eval_data = pd.concat([eval_data,pd.read_pickle(data_dir + data_files[eval_file_inds[i]])])
            eval_features = pd.concat([eval_data,pd.read_pickle(feature_dir + feature_files[eval_file_inds[i]])])


all_inds = np.where(eval_data[target_col].isna() == False)[0]
x_eval = xgb.DMatrix(data = eval_features.iloc[all_inds, :],label=eval_data.loc[all_inds, target_col],feature_names=eval_features.columns)
data = pd.read_pickle(data_dir + data_files[-2])
features = pickle.load(open(feature_dir + feature_files[-2], 'rb'))
#subset any rows in the dataset that don't have the target column populated
all_inds = np.where(data[target_col].isna() == False)[0]
if not len(all_inds) == data.shape[0]:
    features = features.iloc[all_inds, :]
    features = features.reset_index(drop=True)
    data = data.iloc[all_inds, :]
    data = data.reset_index(drop=True)

target_vals = data[target_col]
del data
gc.collect()
# split the data
X_train, X_test, y_train, y_test = train_test_split(features, target_vals, test_size=train_test_prop)
# format the training and test sets
train = xgb.DMatrix(X_train, label=y_train, feature_names=features.columns)
test = xgb.DMatrix(X_test, label=y_test, feature_names=features.columns)

del X_train,X_test,y_train,y_test
gc.collect()

# Build XGboost models
x = XgboostOptimizer()
if params is not None:
    x.update_params(params)

x.fit_full_search(dtrain=train, evals=[(train, 'Train'), (test, "Test")])

# classification model
if min(target_vals)==0 and max(target_vals)==1:
    tmp_preds = x.best_model.predict(x_eval)
    print('iter model auc = ' + str(sklearn.metrics.roc_auc_score(x_eval.get_label(), tmp_preds)))


gc.collect()

# save the best model
file = 'models/MetSIRS4_4hr_Model.p'
pickle.dump(x.best_model, open(file, 'wb'))