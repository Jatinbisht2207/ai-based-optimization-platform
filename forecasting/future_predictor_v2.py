import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data


MODEL_PATH = "models/random_forest_model.pkl"


def generate_daily_timestamps(date):

    start = pd.to_datetime(date)
    return [start + timedelta(minutes=30 * i) for i in range(48)]


def predict_full_day(date):

    model = joblib.load(MODEL_PATH)

    df = load_data()
    df = preprocess_data(df)

    df["hour"] = df["Timestamp"].dt.hour

    timestamps = generate_daily_timestamps(date)

    predictions = []

    for ts in timestamps:

        # ==============================
        # 🔥 KEY FIX: SAMPLE BASED ON HOUR
        # ==============================
        filtered = df[df["hour"] == ts.hour]

        if len(filtered) > 0:
            row = filtered.sample(1).iloc[0].copy()
        else:
            row = df.sample(1).iloc[0].copy()

        # ==============================
        # TIME FEATURES
        # ==============================
        row["hour"] = ts.hour
        row["day"] = ts.day
        row["month"] = ts.month
        row["day_of_week"] = ts.dayofweek

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

        pred = model.predict(X)[0]

        predictions.append({
            "Timestamp": ts,
            "Predicted_Consumption": float(pred)
        })

    return pd.DataFrame(predictions)