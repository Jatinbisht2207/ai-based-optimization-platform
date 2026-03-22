import pandas as pd
import numpy as np
import joblib

from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data


MODEL_PATH = "models/random_forest_model.pkl"


def predict_future_consumption(future_timestamp):

    model = joblib.load(MODEL_PATH)

    df = load_data()
    df = preprocess_data(df)

    future_timestamp = pd.to_datetime(future_timestamp)

    last_row = df.iloc[-1].copy()

    # update time features
    last_row["hour"] = future_timestamp.hour
    last_row["day"] = future_timestamp.day
    last_row["month"] = future_timestamp.month
    last_row["day_of_week"] = future_timestamp.dayofweek

    last_row["hour_sin"] = np.sin(2 * np.pi * future_timestamp.hour / 24)
    last_row["hour_cos"] = np.cos(2 * np.pi * future_timestamp.hour / 24)

    last_row["temp_hour_interaction"] = last_row["Temperature"] * last_row["hour"]

    # remove columns not used in training
    drop_cols = ["Electricity_Consumed", "Timestamp", "Anomaly_Label"]

    for col in drop_cols:
        if col in last_row.index:
            last_row = last_row.drop(col)

    X_future = pd.DataFrame([last_row])

    # FORCE SAME FEATURES AS TRAINING
    X_future = X_future[model.feature_names_in_]

    prediction = model.predict(X_future)[0]

    return prediction