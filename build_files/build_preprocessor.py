from TextMiningMachine.io import get_data
from TextMiningMachine.feature_extraction import DataSetBuilder
import pandas as pd
import pickle
import xgboost as xgb
from nltk.corpus import stopwords

# Query Data (takes about 2 hrs )
sql = "SELECT * FROM [StatisticalModels].[dbo].[InpatientPipelineLabsVitals_20180101_20180701] WHERE AVAILABLE_TIME > '2018-03-01'"
data = get_data('muscedw', sql)


# load the pickled data
#data = pd.read_pickle('data/raw_data.p')


#data.to_csv('data/raw_data.csv')
################################.

# save the raw_data
file = 'data/raw_data.p'
pickle.dump(data, open(file, 'wb'))

# set up the parameters
col_dict = {'cat_cols': ['MUSC R AS IV DEVICE WDL', 'PAT_SEX', 'FINANCIAL_CLASS_NAME',
                        'AGE_GROUP'],
            'cat_from_text_cols':['CPM S16 R INV O2 DEVICE', 'CPM S16 R INV ISOLATION PRECAUTIONS'],
            'zero_imputer_cols':[ 'AGE', 'AVAILABLE_TIME_HOUR', 'AVAILABLE_TIME_MONTH',
                        'AVAILABLE_TIME_DAYOFWEEK', 'AVAILABLE_TIME_DAY','ICUSixMonths', 'PCPSixMonths',
                        'SurgerySixMonths', 'VenilatorSixMonths', 'AdmissionsSixMonths', 'MaxPreviousLOS'
                                 ],
            'imputer_cols':['GLUCOSE, WHOLE BLOOD',
                               'HEMOLYSIS INDEX', 'SODIUM', 'POTASSIUM', 'GLUCOSE', 'CREATININE',
                               'CHLORIDE', 'CALCIUM', 'CO2 CONTENT (BICARBONATE)',
                               'UREA NITROGEN, BLOOD (BUN)', 'ANION GAP', 'HEMATOCRIT',
                               'HEMOGLOBIN', 'PLATELET COUNT', 'RED BLOOD CELL COUNT',
                               'MEAN CORPUSCULAR HEMOGLOBIN', 'MEAN CORPUSCULAR HEMOGLOBIN CONC',
                               'MEAN CORPUSCULAR VOLUME', 'WHITE BLOOD CELL COUNT',
                               'RED CELL DISTRIBUTION WIDTH', 'MEAN PLATELET VOLUME',
                               'ICTERIC INDEX', 'MAGNESIUM', 'NUCLEATED RED BLOOD CELLS',
                               'PHOSPHORUS (PO4)', 'EGFR', 'BILIRUBIN, TOTAL', 'TOTAL PROTEIN',
                               'ALBUMIN', 'ASPARTATE AMINOTRANSFERASE (AST)(SGOT)',
                               'ALKALINE PHOSPHATASE', 'ALANINE AMINOTRANSFERASE (ALT)(SGPT)',
                               'FIO2, ARTERIAL', 'PO2 (CORR), ARTERIAL', 'pH (CORR), ARTERIAL',
                               'BICARB, ARTERIAL', 'PCO2 (CORR), ARTERIAL', 'BASE, ARTERIAL',
                               'O2 SAT, ARTERIAL', 'TOTAL CO2, ARTERIAL',
                               'PT TEMP (CORR), ARTERIAL', 'PROTHROMBIN TIME', 'INR',
                               'NEUTROPHILS ABSOLUTE COUNT', 'MONOCYTES RELATIVE PERCENT',
                               'LYMPHOCYTES ABSOLUTE COUNT', 'NEUTROPHILS RELATIVE PERCENT',
                               'LYMPHOCYTE RELATIVE PERCENT', 'MONOCYTES ABSOLUTE COUNT',
                               'EOSINOPHILS, ABSOLUTE COUNT', 'PULSE', 'PULSE OXIMETRY',
                               'RESPIRATIONS', 'TEMPERATURE',
                               'R MAP',
                               'CPM S16 R AS PAIN RATING (0-10): REST', 'R MAINTENANCE IV VOLUME',
                               'ORAL INTAKE', 'URINE OUTPUT',
                               'CPM S16 R AS SC BRADEN SCORE', 'MUSC R URINE OUTPUT (ML)',
                               'CPM F12 ROW TUBE FEEDING INTAKE (ML) (ADULT, NICU, OB, PEDIATRIC)',
                               'R MAP A-LINE',
                               'R MORSE FALL RISK SCORE', 'MUSC R GENERAL OUTPUT (ML)',
                               'CPM S16 R AS SC GLASGOW COMA SCALE SCORE', 'WEIGHT/SCALE',
                               'R IP FN WEIGHT CHANGE',
                               'MUSC IP CCPOT TOTAL SCORE', 'CPM S16 R AS SC NIPS SCORE',
                               'MUSC IP R AVPU (TRANSFORMED)',
                               'CPM S16 R AS SC ALDRETE SCORE',
                               'CPM S16 R AS CURRENT WEIGHT (GM) (PEDIATRIC)',
                               'CPM S16 R AS SC BRADEN Q SCORE', 'BLOOD PRESSURE (SYSTOLIC)',
                               'BLOOD PRESSURE (DIASTOLIC)',
                               'MUSC R SC PHLEBITIS IV DEVICE (TRANSFORMED)',
                               'MUSC R AS SC INFILTRATION IV DEVICE (TRANSFORMED)',
                               'R ARTERIAL LINE BLOOD PRESSURE (SYSTOLIC)',
                               'R ARTERIAL LINE BLOOD PRESSURE (DIASTOLIC)',
                               'CPM S16 R AS SC RASS (RICHMOND AGITATION-SEDATION SCALE) (TRANSFORMED)',
                               'R MUSC ED WISCONSIN SEDATION SCALE (TRANSFORMED)', 'MetTemp',
                               'MetHR', 'MetRR', 'MetWBC', 'MaxTemp8', 'MaxTemp24', 'MaxTemp48',
                               'MinTemp8', 'MinTemp24', 'MinTemp48', 'MaxHR8', 'MaxHR24', 'MaxHR48',
                               'MinHR8', 'MinHR24', 'MinHR48', 'MaxRR8', 'MaxRR24', 'MaxRR48', 'MinRR8', 'MinRR24',
                               'MinRR48', 'MaxWBC8', 'MaxWBC24', 'MaxWBC48', 'MinWBC8', 'MinWBC24', 'MinWBC48',
                               'DaysSinceLastAdmission'
                            ]}

# data[col_dict.get('imputer_cols')] = data[col_dict.get('imputer_cols')].fillna(-99999.0)
# data[col_dict.get('zero_imputer_cols')] = data[col_dict.get('zero_imputer_cols')].fillna(0.0)
#
# data[col_dict.get('imputer_cols')] = data[col_dict.get('imputer_cols')].convert_objects(convert_numeric=True)
# data[col_dict.get('zero_imputer_cols')] = data[col_dict.get('zero_imputer_cols')].convert_objects(convert_numeric=True)


# Learn the preProcessing
stops = stopwords.words('english')+['stage', 'unspecified', 'hold', 'call', 'defined', 'secondary', 'welcome',
                                    'mention', 'hospital','delivery']
trans = DataSetBuilder(col_dict=col_dict)

trans.params['cat_cols']['min_freq'] = .003
trans.params['cat_from_text_cols']['min_freq'] = .0005

trans.fit(data)


# save the transform
file = 'models/text_cat_transformer.p'
pickle.dump(trans, open(file, 'wb'))


# save the training data
features = xgb.DMatrix(trans.transform(data), feature_names=trans.feature_names)
xgb.DMatrix.save_binary(features, 'data/xgb.features.data')
