import pandas as pd
def preprocess_data(df):
    df.columns = df.columns.str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df['hour'] = df['Timestamp'].dt.hour
    df['day'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df = df.drop(columns=['Timestamp'])
    if 'Anomaly_Label' in df.columns:
        df = df.drop(columns=['Anomaly_Label'])

    print("Preprocessing completed")
    return df