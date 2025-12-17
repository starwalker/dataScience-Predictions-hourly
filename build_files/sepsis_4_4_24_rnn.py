from easy_sql.io import Session as session
from easy_sql.utils import read_sql_text
from sklearn.preprocessing import Imputer, StandardScaler
import numpy as np
from keras import Model
from keras.layers import Dense, Dropout, Input, GaussianNoise, GRU
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.callbacks import EarlyStopping
import pandas as pd
from rnner.metrics import auc_roc
import pickle
np.random.seed(7)
pd.set_option('max_colwidth', 800)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
dsn = 'muscedw'

time_col = ['REPORTING_TIME']
key_col = ['PAT_ENC_CSN_ID']
y_cols = ['MetSIRS4_4hr_24']
x_cols = ['GLUCOSE, WHOLE BLOOD', 'HEMOLYSIS INDEX', 'SODIUM', 'POTASSIUM', 'GLUCOSE', 'CREATININE', 'CHLORIDE',
          'CALCIUM', 'CO2 CONTENT (BICARBONATE)', 'UREA NITROGEN, BLOOD (BUN)', 'ANION GAP', 'HEMATOCRIT', 'HEMOGLOBIN',
          'PLATELET COUNT', 'RED BLOOD CELL COUNT', 'MEAN CORPUSCULAR HEMOGLOBIN', 'MEAN CORPUSCULAR HEMOGLOBIN CONC',
          'MEAN CORPUSCULAR VOLUME', 'WHITE BLOOD CELL COUNT', 'RED CELL DISTRIBUTION WIDTH', 'MEAN PLATELET VOLUME',
          'MAGNESIUM', 'NUCLEATED RED BLOOD CELLS', 'PHOSPHORUS (PO4)', 'EGFR', 'BILIRUBIN, TOTAL', 'TOTAL PROTEIN',
          'ALBUMIN', 'ASPARTATE AMINOTRANSFERASE (AST)(SGOT)', 'ALKALINE PHOSPHATASE',
          'ALANINE AMINOTRANSFERASE (ALT)(SGPT)',
          'FIO2, ARTERIAL', 'PO2 (CORR), ARTERIAL', 'pH (CORR), ARTERIAL', 'BICARB, ARTERIAL', 'PCO2 (CORR), ARTERIAL',
          'BASE, ARTERIAL', 'O2 SAT, ARTERIAL', 'TOTAL CO2, ARTERIAL', 'PT TEMP (CORR), ARTERIAL', 'PROTHROMBIN TIME',
          'INR', 'NEUTROPHILS ABSOLUTE COUNT', 'MONOCYTES RELATIVE PERCENT', 'LYMPHOCYTES ABSOLUTE COUNT',
          'NEUTROPHILS RELATIVE PERCENT', 'LYMPHOCYTE RELATIVE PERCENT', 'MONOCYTES ABSOLUTE COUNT',
          'EOSINOPHILS, ABSOLUTE COUNT', 'BLOOD PRESSURE (SYSTOLIC)', 'BLOOD PRESSURE (DIASTOLIC)',
          'MUSC R SC PHLEBITIS IV DEVICE (TRANSFORMED)', 'MUSC R AS SC INFILTRATION IV DEVICE (TRANSFORMED)',
          'R ARTERIAL LINE BLOOD PRESSURE (SYSTOLIC)', 'R ARTERIAL LINE BLOOD PRESSURE (DIASTOLIC)',
          'CPM S16 R AS SC RASS (RICHMOND AGITATION-SEDATION SCALE) (TRANSFORMED)',
          'R MUSC ED WISCONSIN SEDATION SCALE (TRANSFORMED)', 'MaxWBC8', 'MaxWBC24',
          'MaxWBC48', 'MinWBC8', 'MinWBC24', 'MinWBC48']

# get data
s = session(dsn)
sql = 'select top 1000000 * FROM [StatisticalModels].[dbo].[LabsVitalsHourly]'
print('getting data from', dsn, sql)
data = s.get_data(sql)

# filter out where outcome is None, or row sums are zero
data.dropna(subset=y_cols, inplace=True)
data[y_cols +  key_col] = data[y_cols +  key_col].astype(np.int)
data = data[data.sum(axis=1) > 0]

# fit a scaler and imputer
imp = Imputer()
scale = StandardScaler()
imp.fit(data[x_cols])
scale.fit(np.array(data[x_cols].dropna()))

seq_len = 12
x_dims = len(x_cols)
y_dims = len(y_cols)


# creates and input generator
def input_gen(data, key_col=key_col, time_col=time_col, x_col=x_cols, y_col=y_cols, batch_size = 100, seq_len=seq_len, imputer=imp, scaler=scale):
    keys = np.unique(np.array(data[key_col]))
    n_keys = len(keys)
    print('n_steps:', int(n_keys/batch_size))
    key_iter = iter(keys)
    while True:
        print('generating data  ...')
        x_list = []
        y_list = []
        temp_keys = [next(key_iter) for _ in range(batch_size)]
        for key in temp_keys:
            temp_index = np.array(data[key_col] == key)
            keep_index = [i for i in range(data.shape[0]) if temp_index[i]]
            temp_data = data.iloc[keep_index]
            temp_data.set_index(time_col)
            temp_data.sort_index(axis=0, inplace=True)
            n_chunks = int(temp_data.shape[0]/seq_len)
            if n_chunks > 0:
                temp_x = temp_data[x_cols]
                if imputer is not None:
                    temp_x = imputer.transform(temp_x)
                if scaler is not None:
                    temp_x = scaler.transform(temp_x)
                temp_x = np.array(temp_x)[-n_chunks * seq_len:]
                temp_y = np.array(temp_data[y_cols])[-n_chunks * seq_len:]
                temp_x_reshaped = np.reshape(temp_x, (n_chunks, seq_len, temp_x.shape[1]))
                temp_y_reshaped = np.reshape(temp_y, (n_chunks, seq_len, temp_y.shape[1]))
                x_list.append(temp_x_reshaped)
                y_list.append(temp_y_reshaped)
        x_train = np.concatenate(x_list)
        y_train = np.concatenate(y_list)
        yield x_train, y_train


esm = EarlyStopping(patience=1)
# trains a multiple output model
inputs = Input(shape=(seq_len, x_dims))
d1 = Dense(10, activation='relu')(inputs)
noise = GaussianNoise(1.0)(d1)
gru1 = GRU(12, return_sequences=True)(noise)
dropout = Dropout(.2)(gru1)
outputs = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_roc])


batch_size = 5000
total_epochs = 10
sub_epocs = 5
w = None
for i in range(total_epochs):
    print('epochs: ', i, 'of ', total_epochs)
    g = input_gen(data, key_col=key_col, time_col=time_col, x_col=x_cols, y_col=y_cols, batch_size = batch_size,
                  seq_len=seq_len, imputer=imp, scaler=scale)
    if i > 0:
        model.set_weights(w)
    while True:
        try:
            x, y = next(g)
            model.fit(x, y, epochs=sub_epocs, callbacks=[esm], validation_data=(x_test,y_test), batch_size=100, shuffle=True)
            w = model.get_weights()
        except:
            break
