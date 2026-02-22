import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.config import TARGET_COLUMN, TEST_SIZE

class ARIMAModel:
    def train(self, df):

        # Only keep target column
        series = df[TARGET_COLUMN]
        split_index = int(len(series) * (1 - TEST_SIZE))
        train = series.iloc[:split_index]
        test = series.iloc[split_index:]
        # Fit ARIMA
        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()
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