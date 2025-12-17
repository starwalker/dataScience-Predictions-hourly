from Hourly_Predictions.io import BatchPuller
from Hourly_Predictions.model_building import OutOfMemoryNN
from keras import Model
from keras.layers import Dense, Dropout, Input, GaussianNoise
from keras.callbacks import EarlyStopping
from rnner.metrics import auc_roc

model_file_path = 'models\mews_nn.bundle'
data_dir = 'data\lv'


y_cols = ['MetMEWS4_4', 'MetMEWS4_8', 'MetMEWS4_24', 'MetMEWS4_48']
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

# define parameters for training the model
epochs = 2
n_passes = 2
batch_size = 1000
split = .75

# define the model
inputs = Input(shape=(len(x_cols),))
noise = GaussianNoise(1.0)(inputs)
d1 = Dense(100, activation='relu')(noise)
dropout = Dropout(.5)(d1)
outputs = Dense(len(y_cols), activation='sigmoid')(dropout)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', auc_roc])

#
omn = OutOfMemoryNN(data_dir=data_dir, model=model, epochs=epochs, batch_size=batch_size, n_passes=n_passes,
                      x_cols=x_cols, y_cols=y_cols, split=split)
# fit the scaler and imputer
omn.fit_scaler_imputer()
omn.build_model()
b = omn.bundle()
b.save_model()

# test methods
g = omn.train_test_gen()
x_train, y_train, x_test, y_test = next(g)
preds = b.predict(features=x_train)
b.fit_deciles()
decs = b.predict_deciles(features=x_train)
