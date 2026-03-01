import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE


class RandomForestModel:

    def train_or_load(self, df, train_mode=True):

        model_path = "models/random_forest_model.pkl"
        os.makedirs("models", exist_ok=True)

        # Ensure chronological order
        df = df.sort_values("Timestamp")

        split_index = int(len(df) * (1 - TEST_SIZE))

        train = df.iloc[:split_index]
        test = df.iloc[split_index:]

        # Timestamp automatically excluded (not numeric)
        X_train = train.select_dtypes(include=["number"]).drop(columns=[TARGET_COLUMN])
        y_train = train[TARGET_COLUMN]

        X_test = test.select_dtypes(include=["number"]).drop(columns=[TARGET_COLUMN])
        y_test = test[TARGET_COLUMN]

        if train_mode or not os.path.exists(model_path):
            print("Training Random Forest model...")
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            joblib.dump(model, model_path)
            print("Random Forest model saved.")
        else:
            print("Loading existing Random Forest model...")
            model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        metrics = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred))
        }

        # Preserve Timestamp in predictions
        predictions_df = pd.DataFrame({
            "Timestamp": test["Timestamp"].values,
            "Actual": y_test.values,
            "Predicted": y_pred
        })

        return model, metrics, predictions_df