import os
import pandas as pd
import numpy as np
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.config import TARGET_COLUMN, TEST_SIZE

class SARIMAModel:
    def train(self, df):
        model_path = "models/sarima_model.pkl"

        series = df[TARGET_COLUMN]
        split_index = int(len(series) * (1 - TEST_SIZE))
        train = series.iloc[:split_index]
        test = series.iloc[split_index:]
        # Create models folder
        os.makedirs("models", exist_ok=True)
        # If model exists → load it
        if os.path.exists(model_path):
            print("Loading existing SARIMA model...")
            model_fit = joblib.load(model_path)
        else:
            print("Training SARIMA model...")
            model = SARIMAX(
                train,
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, 48),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)
            joblib.dump(model_fit, model_path)
            print("SARIMA model saved.")
        forecast = model_fit.forecast(steps=len(test))
        metrics = {
            "MAE": float(mean_absolute_error(test, forecast)),
            "RMSE": float(np.sqrt(mean_squared_error(test, forecast))),
            "R2": float(r2_score(test, forecast))
        }
        predictions_df = pd.DataFrame({
            "Actual": test.values,
            "Predicted": forecast.values
        })
        return model_fit, metrics, predictions_df