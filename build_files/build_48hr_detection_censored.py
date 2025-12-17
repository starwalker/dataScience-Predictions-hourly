import pickle
import xgboost as xgb
import gc
import pandas as pd
import numpy as np
from TextMiningMachine.splitting import TrainTestDateSplitter
from TextMiningMachine.xgboost_optimizer import XgboostOptimizer
from TextMiningMachine.utils import generate_column_from_search
import math
import datetime
import sklearn
gc.collect()
import datetime

target_col = 'MetSIRS48'
date_col = 'CENSUS_DATE'
# load raw data
data = pd.read_pickle('data/raw_data.p')

data = data.loc[data['NoSIRS12']==1,:]
data[date_col] = data['AVAILABLE_TIME'].dt.date
data = data.reset_index()
# load transform
with open( 'models/text_cat_transformer.p', 'rb') as f:
    trans = pickle.load(f)


features = xgb.DMatrix(trans.transform(data), feature_names=trans.feature_names)
# load preproccessed features
#features = xgb.DMatrix( 'data/xgb.features.data')
features.feature_names = trans.feature_names

cut_date = datetime.date(2018, 6, 1)


# cut by date
s = TrainTestDateSplitter()
test, train, y_test, y_train = s.split(data, target_col, date_col, cut_date, features=features)


#w Build XGboost models
x = XgboostOptimizer()

## Perform a iterative random grid search, where each iteration samples
## values for the parameters between the LHS and RHS specified

# update features related to random search algorithm
x.update_params({'objective': 'binary:logistic',
                 'eval_metric': 'auc',
                 'max_over_fit': .03,
                 'eta': np.arange(.01, .25, .01),
                 'max_depth': range(1, 12),
                 'min_child_weight': range(5000, 25000, 25),
                 'num_boost_rounds': 50,
                 'early_stopping_rounds': 2})

#call random search algorithm
x.fit_full_search(dtrain = train, evals=[(train, 'Train'), (test, "Test")])


# extract the best model
model = x.best_model
model.feature_names = trans.feature_names

importance_frame = model.get_score(importance_type='gain')
importance_frame = pd.DataFrame({'Importance': list(importance_frame.values()), 'Feature': list(importance_frame.keys())})
importance_frame.sort_values(by = 'Importance', inplace=True, ascending=[False])
importance_frame = importance_frame.reset_index(drop=True)
importance_frame.head(50).sort_values(by = 'Importance').plot(x ='Feature', y='Importance' ,kind='barh',legend=False)
importance_frame

# save the optimal params
file = 'models/optimal_sepsis_48hrs_params.p'
pickle.dump(x.xgb_params_best, open(file, 'wb'))

# save the best model
file = 'models/sepsis_48hrs.p'
pickle.dump(model, open(file, 'wb'))
