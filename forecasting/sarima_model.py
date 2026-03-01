import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.config import TARGET_COLUMN, TEST_SIZE


class SARIMAModel:
    def train(self, df):
        # Ensure chronological order
        df = df.sort_values("Timestamp")

        split_index = int(len(df) * (1 - TEST_SIZE))

        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        y_train = train[TARGET_COLUMN]
        y_test = test[TARGET_COLUMN]

        print("Training SARIMA model...")

        # ✅ Correct seasonal order (daily seasonality: 48 steps)
        model = SARIMAX(
            y_train,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, 48),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted_model = model.fit(disp=False)

        forecast = fitted_model.forecast(steps=len(test))

        metrics = {
            "MAE": float(mean_absolute_error(y_test, forecast)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, forecast))),
            "R2": float(r2_score(y_test, forecast))
        }

        # Preserve Timestamp
        predictions_df = pd.DataFrame({
            "Timestamp": test["Timestamp"].values,
            "Actual": y_test.values,
            "Predicted": forecast.values
        })

        return fitted_model, metrics, predictions_df