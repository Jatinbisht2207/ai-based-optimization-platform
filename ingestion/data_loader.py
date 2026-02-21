import pandas as pd
from utils.config import DATA_PATH
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        print("Data loaded successfully")
        return df
    except Exception as e:
        print(" Error loading data:", e)
        raise