import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(df):

    encoders = {}

    # find categorical columns automatically
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col].astype(str))

        encoders[col] = le

    columns = df.columns.tolist()

    scaler = MinMaxScaler(feature_range=(-1,1))

    data_scaled = scaler.fit_transform(df)

    return data_scaled, scaler, columns, encoders
