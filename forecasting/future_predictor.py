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
    # 🔥 FIX: pick similar historical row
    # ==============================
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    similar_rows = df[
        (df["hour"] == future_timestamp.hour) &
        (df["day_of_week"] == future_timestamp.dayofweek)
    ]

    if len(similar_rows) > 0:
        base_row = similar_rows.sample(1).iloc[0].copy()  # random similar pattern
    else:
        base_row = df.sample(1).iloc[0].copy()  # fallback random row

    # ==============================
    # UPDATE TIME FEATURES
    # ==============================
    base_row["hour"] = future_timestamp.hour
    base_row["day"] = future_timestamp.day
    base_row["month"] = future_timestamp.month
    base_row["day_of_week"] = future_timestamp.dayofweek

    base_row["hour_sin"] = np.sin(2 * np.pi * future_timestamp.hour / 24)
    base_row["hour_cos"] = np.cos(2 * np.pi * future_timestamp.hour / 24)

    base_row["temp_hour_interaction"] = base_row["Temperature"] * base_row["hour"]

    # ==============================
    # REMOVE UNUSED COLUMNS
    # ==============================
    drop_cols = ["Electricity_Consumed", "Timestamp", "Anomaly_Label"]

    for col in drop_cols:
        if col in base_row.index:
            base_row = base_row.drop(col)

    X_future = pd.DataFrame([base_row])

    # ==============================
    # MATCH TRAINING FEATURES
    # ==============================
    X_future = X_future[model.feature_names_in_]

    prediction = model.predict(X_future)[0]

    return float(prediction)