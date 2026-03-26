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

    # ==============================
    # 🔥 ADD TIME FEATURES TO DATA
    # ==============================
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    timestamps = generate_daily_timestamps(date)

    predictions = []

    for ts in timestamps:

        # ==============================
        # 🔥 FIX: pick similar historical row
        # ==============================
        similar_rows = df[
            (df["hour"] == ts.hour) &
            (df["day_of_week"] == ts.dayofweek)
        ]

        if len(similar_rows) > 0:
            row = similar_rows.sample(1).iloc[0].copy()
        else:
            row = df.sample(1).iloc[0].copy()

        # ==============================
        # UPDATE TIME FEATURES
        # ==============================
        row["hour"] = ts.hour
        row["day"] = ts.day
        row["month"] = ts.month
        row["day_of_week"] = ts.dayofweek

        row["hour_sin"] = np.sin(2 * np.pi * ts.hour / 24)
        row["hour_cos"] = np.cos(2 * np.pi * ts.hour / 24)

        row["temp_hour_interaction"] = row["Temperature"] * row["hour"]

        # ==============================
        # REMOVE UNUSED COLUMNS
        # ==============================
        drop_cols = ["Electricity_Consumed", "Timestamp", "Anomaly_Label"]

        for col in drop_cols:
            if col in row.index:
                row = row.drop(col)

        X = pd.DataFrame([row])

        # ==============================
        # MATCH TRAINING FEATURES
        # ==============================
        X = X[model.feature_names_in_]

        pred = model.predict(X)[0]

        predictions.append({
            "Timestamp": ts,
            "Predicted_Consumption": float(pred)
        })

    result_df = pd.DataFrame(predictions)

    return result_df