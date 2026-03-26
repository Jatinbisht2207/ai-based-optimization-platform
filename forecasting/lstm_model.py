# ============================================
# 🔥 FINAL LSTM PREDICTION MODULE
# (FIXED - CHANGE BASED + NEW SCALERS)
# ============================================

import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data


# ==============================
# PATHS
# ==============================
MODEL_PATH = "models/lstm_model.h5"
FEATURE_SCALER_PATH = "models/lstm_feature_scaler.pkl"
TARGET_SCALER_PATH = "models/lstm_target_scaler.pkl"

TIME_STEPS = 48


# ==============================
# MAIN FUNCTION
# ==============================
def predict_lstm():

    # Load model & scalers
    model = load_model(MODEL_PATH, compile=False)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler = joblib.load(TARGET_SCALER_PATH)

    # Load data
    df = load_data()
    df = preprocess_data(df)

    # Time features (must match training)
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek

    # Feature set (must match training EXACTLY)
    features = [
        "Electricity_Consumed",
        "Temperature",
        "Humidity",
        "Wind_Speed",
        "hour",
        "day_of_week",
        "Avg_Past_Consumption"
    ]

    data = df[features].copy()

    # ==============================
    # SCALE INPUT
    # ==============================
    scaled_data = feature_scaler.transform(data)

    # ==============================
    # CREATE SEQUENCES
    # ==============================
    X = []
    for i in range(TIME_STEPS, len(scaled_data)):
        X.append(scaled_data[i - TIME_STEPS:i])

    X = np.array(X)

    # ==============================
    # MODEL PREDICTION (CHANGE)
    # ==============================
    preds = model.predict(X)

    # Inverse scaling (target change)
    pred_change = target_scaler.inverse_transform(preds).flatten()

    # ==============================
    # FINAL VALUE = BASE + CHANGE
    # ==============================
    base_values = data.iloc[TIME_STEPS:]["Electricity_Consumed"].values
    final_predictions = base_values + pred_change

    # ==============================
    # PREPARE OUTPUT
    # ==============================
    df = df.iloc[TIME_STEPS:].copy()
    df["Predicted"] = final_predictions

    return df