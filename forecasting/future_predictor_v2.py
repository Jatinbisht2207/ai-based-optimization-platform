import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data


MODEL_PATH = "models/random_forest_model.pkl"


def generate_daily_timestamps(date):

    start = pd.to_datetime(date)

    timestamps = []

    for i in range(48):
        timestamps.append(start + timedelta(minutes=30 * i))

    return timestamps


def predict_full_day(date):

    model = joblib.load(MODEL_PATH)

    df = load_data()
    df = preprocess_data(df)

    base_row = df.iloc[-1].copy()

    timestamps = generate_daily_timestamps(date)

    predictions = []

    for ts in timestamps:

        row = base_row.copy()

        row["hour"] = ts.hour
        row["day"] = ts.day
        row["month"] = ts.month
        row["day_of_week"] = ts.dayofweek

        row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)

        row["temp_hour_interaction"] = row["Temperature"] * row["hour"]

        drop_cols = ["Electricity_Consumed", "Timestamp", "Anomaly_Label"]

        for col in drop_cols:
            if col in row.index:
                row = row.drop(col)

        X = pd.DataFrame([row])

        # FORCE SAME FEATURES AS TRAINING
        X = X[model.feature_names_in_]

        pred = model.predict(X)[0]

        predictions.append({
            "Timestamp": ts,
            "Predicted_Consumption": pred
        })

    result_df = pd.DataFrame(predictions)

    return result_df