import pandas as pd
import numpy as np
from utils.config import TARGET_COLUMN

def preprocess_data(df):

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)

    # Sort chronologically
    df = df.sort_values('Timestamp')

    # Time features
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['day_of_week'] = df['Timestamp'].dt.dayofweek

    # 🔥 Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # 🔥 Lag features
    df['lag_1'] = df[TARGET_COLUMN].shift(1)
    df['lag_2'] = df[TARGET_COLUMN].shift(2)
    df['lag_3'] = df[TARGET_COLUMN].shift(3)
    df['lag_24'] = df[TARGET_COLUMN].shift(24)

    # 🔥 Rolling statistics
    df['rolling_mean_3'] = df[TARGET_COLUMN].shift(1).rolling(window=3).mean()
    df['rolling_mean_24'] = df[TARGET_COLUMN].shift(1).rolling(window=24).mean()

    df['rolling_std_3'] = df[TARGET_COLUMN].shift(1).rolling(window=3).std()
    df['rolling_std_24'] = df[TARGET_COLUMN].shift(1).rolling(window=24).std()

    # 🔥 Interaction feature
    df['temp_hour_interaction'] = df['Temperature'] * df['hour_sin']

    # Drop timestamp
    df = df.drop(columns=['Timestamp'])

    # Remove NaN rows created by lags/rolling
    df = df.dropna().reset_index(drop=True)

    print("Preprocessing upgraded with advanced time-series features.")

    return df