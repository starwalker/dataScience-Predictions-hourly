if __name__ == '__main__':
    import pickle
    import pandas as pd
    import numpy as np
    import xgboost as xgb



    with open('data/raw_data.p', 'rb') as f:
        data = pickle.load(f)

    # load transform
    with open('models/sepsis_48hrs.p', 'rb') as f:
        model = pickle.load(f)

    # load preproccessed features
    features = xgb.DMatrix('data/xgb.features.data')
    features.feature_names = model.feature_names
    preds = model.predict(features)

    data['AVAILABLE_DATE'] = data['AVAILABLE_TIME'].dt.strftime('%Y-%m-%d')

    output = pd.concat((data[['PAT_ID', 'PAT_ENC_CSN_ID']],pd.Series(preds),data[['AGE','AVAILABLE_DATE','AVAILABLE_TIME_HOUR','AVAILABLE_TIME_MONTH','AVAILABLE_TIME_DAYOFWEEK', 'AVAILABLE_TIME_DAY','MetSIRS8',
       'MetSIRS24', 'MetSIRS48','NoSIRS12']]),axis=1)
    output.columns.values[2] = 'Predictions'


    output.to_csv('data/Sepsis48SimulationData.csv',index=False)