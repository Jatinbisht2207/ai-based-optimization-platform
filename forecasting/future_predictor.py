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

    # ==============================
    # 🔥 KEY FIX: USE HISTORICAL SAMPLE
    # ==============================
    df["hour"] = df["Timestamp"].dt.hour

    filtered = df[df["hour"] == future_timestamp.hour]

    if len(filtered) > 0:
        row = filtered.sample(1).iloc[0].copy()
    else:
        row = df.sample(1).iloc[0].copy()

    # ==============================
    # TIME FEATURES
    # ==============================
    row["hour"] = future_timestamp.hour
    row["day"] = future_timestamp.day
    row["month"] = future_timestamp.month
    row["day_of_week"] = future_timestamp.dayofweek

    row["hour_sin"] = np.sin(2 * np.pi * row["hour"] / 24)
    row["hour_cos"] = np.cos(2 * np.pi * row["hour"] / 24)

    row["temp_hour_interaction"] = row["Temperature"] * row["hour"]

    # ==============================
    # CLEAN FEATURES
    # ==============================
    drop_cols = ["Electricity_Consumed", "Timestamp", "Anomaly_Label"]

    for col in drop_cols:
        if col in row.index:
            row = row.drop(col)

    X = pd.DataFrame([row])
    X = X[model.feature_names_in_]

    prediction = model.predict(X)[0]

    return float(prediction)