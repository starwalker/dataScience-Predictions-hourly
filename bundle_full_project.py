from TextMiningMachine.io import get_data
from TextMiningMachine.feature_extraction import DataSetBuilder
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
from TextMiningMachine.bundle import XGBModelBundle, XGBModelGroupBundle

data = pd.read_pickle('data/raw_data.p')
trans = pickle.load(open('dsvm_build_files/text_cat_transformer.p','rb'))
date_col = 'REPORTING_TIME'


features = sparse.load_npz('data/features.npz')

keep_cols = ['PAT_ID', 'PAT_ENC_CSN_ID', 'REPORTING_TIME']
target_cols = ['MetSIRS4_4hr_8', 'MetSIRS4_4hr_24', 'MetSIRS4_4hr_48']

model_path = 'models/xgb/'
for target_col in target_cols:
    model = pickle.load(open(model_path+target_col+'_Model.p','rb'))

    mb = XGBModelBundle(target_col)
    mb.update({'trans': trans,
               'model': model,
               'date_based_split': True,
               'cut_date': datetime.date(2017, 1, 19),
               'date_col' : 'REPORTING_TIME'})
    preds = mb.generate_predictions(train)
    mb.fit_deciles(list(preds))
    model_file_name = model_path + target_col + '_Bundle.p'
    mb.save(model_file_name)


target_cols = ['MetMEWS4_8', 'MetMEWS4_24', 'MetMEWS4_48']

model_path = 'models/xgb/'
for target_col in target_cols:
    model = pickle.load(open(model_path+target_col+'_Model.p','rb'))

    mb = XGBModelBundle(target_col)
    mb.update({'trans': trans,
               'model': model,
               'date_based_split': True,
               'cut_date': datetime.date(2018, 4, 7),
               'date_col' : 'REPORTING_TIME'})
    data_transformations = ['keep_inds = np.where(data[date_col]>datetime.datetime(2017, 5, 1))[0]',
                            'data = data.iloc[keep_inds,:]',
                            'features = features[keep_inds,:]',
                            'data = data.reset_index(drop=True)',
                            'gc.collect()']
    mb.data_transformations = data_transformations
    preds = mb.generate_predictions(train)
    mb.fit_deciles(list(preds))
    model_file_name = model_path + target_col + '_Bundle.p'
    mb.save(model_file_name)



target_col = 'MetMEWS4_24'
model = pickle.load(open(model_path+target_col+'_Model.p','rb'))

mb = XGBModelBundle(target_col)
mb.update({'trans': trans,
           'model': model,
           'date_based_split': True,
           'cut_date': datetime.date(2018, 4, 7),
           'date_col' : 'REPORTING_TIME'})
preds = mb.generate_predictions(test)
mb.fit_deciles(list(preds))
model_file_name = model_path + target_col + '_Bundle.p'
mb.save(model_file_name)





target_col = 'MetSIRS3_4hr_24'
model = pickle.load(open(model_path+target_col+'Uniform6_Model.p','rb'))

mb = XGBModelBundle(target_col)
mb.update({'trans': trans,
           'model': model,
           'date_based_split': True,
           'cut_date': datetime.date(2017, 1, 19),
           'date_col' : 'REPORTING_TIME'})
preds = mb.generate_predictions(features)
mb.fit_deciles(list(preds))
model_file_name = model_path + target_col + '_Bundle.p'
mb.save(model_file_name)



target_col = 'MetSIRS4_4hr_24'
model = pickle.load(open(model_path+target_col+'Uniform6_Model.p','rb'))

mb = XGBModelBundle(target_col)
mb.update({'trans': trans,
           'model': model,
           'date_based_split': True,
           'cut_date': datetime.date(2017, 1, 19),
           'date_col' : 'REPORTING_TIME'})
mb.data_transformations = data_transformations
preds = mb.generate_predictions(features)
mb.fit_deciles(list(preds))
model_file_name = model_path + target_col + '_Bundle.p'
mb.save(model_file_name)



target_col = 'MetSIRSdttm_24'
model = pickle.load(open(model_path+target_col+'_Model.p','rb'))

mb = XGBModelBundle(target_col)
mb.update({'trans': trans,
           'model': model,
           'date_based_split': True,
           'cut_date': datetime.date(2017, 1, 19),
           'date_col' : 'REPORTING_TIME'})
preds = mb.generate_predictions(train)
mb.fit_deciles(list(preds))
model_file_name = model_path + target_col + '_Bundle.p'
mb.save(model_file_name)







from TextMiningMachine.bundle import XGBModelGroupBundle

all_targets = ['MetSIRS3_4hr_24', 'MetSIRS4_4hr_24', 'MetMEWS4_24','MetSIRSdttm_24']

mgb = XGBModelGroupBundle(model_group_name='Hourly_Predictions')
model_filepath_list = [model_path + target_col + '_Bundle.p' for target_col in all_targets]
mgb.set_models_from_filepath(model_filepath_list)

### write_table_key will be generated from the pull table in the future when tom adds in the functionality
mgb.write_table_key = 'PAT_ENC_CSN_ID'
mgb.set_valid_prediction_days(1/24)

model_dict = mgb.model_dict
for target in all_targets:
    model_dict[target] = model_dict.pop(target+'_Bundle')
mgb.trans_model_dict['Transform0'] = model_dict
mgb.model_dict = model_dict
mgb.save('Hourly_Predictions_Bundle.p')

#models_to_build = ['MetMEWS4_8', 'MetMEWS4_24', 'MetMEWS4_48']
models_to_build = ['MetMEWS4_24']
## Here is the call to generate all of the model markdowns. Refer to the function itself to see other paramaeter options
import os


# mgb = pickle.load(open('Hourly_Predictions_bundle.p','rb'))
# mgb.num_top_features = 16
# mgb.generate_model_markdowns(project_path=os.getcwd(), model_list=models_to_build,weave_markdown = False)

#target_cols = ['MetSIRS3_4hr_24', 'MetSIRS4_4hr_24']
for target in models_to_build:
    os.system('pweave -f md2html -o ' + os.getcwd() +"\\markdowns/"+target+'.html'+ ' ' + os.getcwd() + "\\markdowns/"+target+'.pmd')

