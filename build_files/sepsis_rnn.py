from easy_sql.io import Session as session
from easy_sql.utils import read_sql_text
from sklearn.preprocessing import Imputer, StandardScaler
import numpy as np
from keras import Model
from keras.layers import Dense, Dropout, Input, GaussianNoise
from sklearn.metrics import roc_auc_score, accuracy_score
from keras.callbacks import EarlyStopping
import pandas as pd
import pickle
np.random.seed(7)
pd.set_option('max_colwidth', 800)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
dsn = 'muscedw'
sql_file_path = 'data/sepsis_NN.sql'
model_file_path = 'models/sepsis_nn.bundle'

# get data
s = session(dsn)
sql = read_sql_text(sql_file_path)
print('getting data from', dsn, sql)
data = s.get_data(sql)

time_col = 'AVAILABLE_TIME'
id_col = 'PAT_ENC_CSN_ID'
y_cols = ['MetSIRS4_8', 'MetSIRS4_24', 'MetSIRS4_48', 'MetSIRS4_4hr_8', 'MetSIRS4_4hr_24', 'MetSIRS4_4hr_48', 'MS870',
          'MS871', 'MS872', 'Sepsis_DRG']
non_x_cols = [time_col] + [id_col] + y_cols
x_cols = [col for col in data.columns if col not in non_x_cols]

# initialize imputer and scaler
imp = Imputer(strategy='median')
scaler = StandardScaler()

def input_fun(data, fit=False):
    # replace missing values with imputed ones
    temp_data = data[x_cols].replace(-99999, np.nan)
    if fit:
        print('fitting imputer and scaler')
        temp_data_imp = imp.fit_transform(temp_data)
        features = scaler.fit_transform(temp_data_imp)
    else:
        temp_data_imp = imp.transform(temp_data)
        features = scaler.transform(temp_data_imp)
    return features


# extract features
x = input_fun(data, fit=True)
y = np.array(data[y_cols])

# split the trainig and test data
split = .75
cut = int(split * x.shape[0])
x_train = x[:cut, :]
y_train = y[:cut, :]
x_test = x[cut:, :]
y_test = y[cut:, :]

n_input_dims = x_train.shape[1]
n_outputs = y_train.shape[1]

esm = EarlyStopping(patience=2)
# trains a multiple output model
inputs = Input(shape=(n_input_dims, ))
noise = GaussianNoise(1.5)(inputs)
d1 = Dense(200, activation='relu')(noise)
dropout = Dropout(.5)(d1)
outputs = Dense(n_outputs, activation='sigmoid')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, batch_size=1000, validation_data=(x_test, y_test), verbose=2, callbacks=[esm],
          shuffle=True)


# generate predictions
preds_train = model.predict(x_train)
preds_test = model.predict(x_test)

# prints auc off all the output nodes
perf = []
for i in range(n_outputs):
    test_auc = roc_auc_score(y_test[:, i], preds_test[:, i])
    train_auc = roc_auc_score(y_train[:, i], preds_train[:, i])
    perf.append([y_cols[i] + ' train auc: ' + str(train_auc) + ' test auc: ' + str(test_auc)])

print(perf)

import pickle
from rnner.bundle import Bundle
b = Bundle(model=model, scaler=scaler, imputer=imp, data=data, x_cols=x_cols)
b.data = None
pickle.dump(b, open(model_file_path, 'wb'))


# test load
with open(model_file_path, 'rb') as file:
    model_loaded = pickle.load(file)
p = model_loaded.predict(data.iloc[0:10])
if p.shape[0] == 10:
    print('serialization and prediction test completed')