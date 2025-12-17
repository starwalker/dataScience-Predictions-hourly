import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, Imputer
import os
from rnner.bundle import Bundle

model_file_path = 'models\mews_nn.bundle'
data_dir = 'C:\pyCharm\data\hourly_labs_vitials'
data_file_paths = [data_dir + '\\' + j for i, j in enumerate(os.listdir(data_dir))]
train_size = .6
output_cols = ['MetMEWS4_4', 'MetMEWS4_8', 'MetMEWS4_24', 'MetMEWS4_48']
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
n_eval_files = 1
n_files = len(data_file_paths)

# create a final eval holdout set
final_eval_paths = data_file_paths[-n_eval_files:]
train_test_paths = data_file_paths[:(n_files - n_eval_files)]
n_train_files = len(train_test_paths)

# fit the scaler and imputer
with open(train_test_paths[0], 'rb') as f:
    data = pickle.load(f)
imp = Imputer()
scale = StandardScaler()
imp.fit(data[x_cols])
scale.fit(np.array(data[x_cols].dropna()))


# setup a data generator
def train_test_gen(file_paths=train_test_paths, train_size=train_size, scaler=scale, imputer=imp):
    import numpy as np
    import pandas as pd
    np.random.seed(2012)
    for path in file_paths:
        print('loading ...')
        with open(path, 'rb') as f:
            data = pickle.load(f)
        data = data[x_cols + output_cols]
        data.dropna(subset=output_cols, inplace=True)
        data[output_cols] = data[output_cols].astype(np.int)
        data = data.select_dtypes(exclude=object)
        n_obs = data.shape[0]
        n_train_obs = int(train_size * n_obs)
        print('splitting, train: ', n_train_obs, 'n_test_obs: ', n_obs - n_train_obs)
        train_index = np.random.choice(range(n_obs), n_train_obs, replace=False)
        test_index = np.array(list(range(n_obs)))[np.isin(range(n_obs), train_index, invert=True)]
        print('applying imputer and scaler ... ')
        x_train = scaler.transform(imputer.transform(data.iloc[train_index][x_cols]))
        y_train = data.iloc[train_index][output_cols]
        x_test = scaler.transform(imputer.transform(data.iloc[test_index][x_cols]))
        y_test = np.array(data.iloc[test_index][output_cols])
        y_train = np.array(data.iloc[train_index][output_cols])
        yield x_train, y_train, x_test, y_test


# set up a training function
def build_model(train_test_gen=train_test_gen, n_passes=10, n_input_dims=len(x_cols), n_outputs=len(output_cols), n_train_files=n_train_files):
    from keras import Model
    from keras.layers import Dense, Dropout, Input, GaussianNoise
    from keras.callbacks import EarlyStopping
    from rnner.metrics import auc_roc
    inputs = Input(shape=(n_input_dims,))
    noise = GaussianNoise(1.0)(inputs)
    d1 = Dense(100, activation='relu')(noise)
    dropout = Dropout(.5)(d1)
    outputs = Dense(n_outputs, activation='sigmoid')(dropout)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_roc])
    esm = EarlyStopping(patience=1)
    for j in range(n_passes):
        i = 0
        gen = train_test_gen()
        while i < n_train_files:
            x_train, y_train, x_test, y_test = next(gen)
            print('pass: ', j, 'batch: ', i)
            if i > 0:
                model.set_weights(weights=weights)
            # trains a multiple output model
            model.fit(x_train, y_train, epochs=50, batch_size=2000, validation_data=(x_test, y_test), verbose=2,
                      callbacks=[esm],
                      shuffle=True)
            weights = model.get_weights()
            i += 1
    return model


# build model
model = build_model()

# save output as bundled object
b = Bundle(model=model, scaler=scale, imputer=imp, x_cols=x_cols, output_cols=output_cols)
pickle.dump(b, open(model_file_path, 'wb'))

