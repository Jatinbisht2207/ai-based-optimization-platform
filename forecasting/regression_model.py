import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

class RegressionModel:
    def train_or_load(self, df, train_mode=True):
        model_path = "models/regression_model.pkl"
        os.makedirs("models", exist_ok=True)
        df = df.sort_index()
        split_index = int(len(df) * (1 - TEST_SIZE))
        train = df.iloc[:split_index]
        test = df.iloc[split_index:]
        X_train = train.select_dtypes(include=["number"]).drop(columns=[TARGET_COLUMN])
        y_train = train[TARGET_COLUMN]
        X_test = test.select_dtypes(include=["number"]).drop(columns=[TARGET_COLUMN])
        y_test = test[TARGET_COLUMN]
        if train_mode or not os.path.exists(model_path):
            print("Training Regression model...")
            model = LinearRegression()
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            print("Regression model saved.")
        else:
            print("Loading existing Regression model...")
            model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred))
        }

        predictions_df = pd.DataFrame({
            "Actual": y_test.values,
            "Predicted": y_pred
        })
        return model, metrics, predictions_df