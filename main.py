import os
import json
import pandas as pd
from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data
from forecasting.regression_model import RegressionModel
from forecasting.random_forest_model import RandomForestModel
from forecasting.arima_model import ARIMAModel
from forecasting.sarima_model import SARIMAModel
from forecasting.future_predictor import predict_future_consumption
from forecasting.future_predictor_v2 import predict_full_day
from anomaly_detection.residual_detector import ResidualAnomalyDetector
from optimization.rule_engine import EnergyOptimizationEngine
from optimization.optimizer import EnergyOptimizer
from explainability.shap_explainer import RandomForestExplainer


# ======================================================
# CONFIG
# ======================================================

TRAIN_MODE = False
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_predictions(df, filename):
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)


# ======================================================
# MAIN PIPELINE
# ======================================================

def main():

    print("\n Starting Smart Energy Analytics System\n")

    # --------------------------------------------------
    # 1️⃣ Load + Preprocess
    # --------------------------------------------------

    df = load_data()
    df = preprocess_data(df)

    model_metrics = {}

    # --------------------------------------------------
    # 2️⃣ Regression
    # --------------------------------------------------

    regression = RegressionModel()
    reg_model, reg_metrics, reg_predictions = regression.train_or_load(df, TRAIN_MODE)

    print("Regression Metrics:", reg_metrics)
    save_predictions(reg_predictions, "regression_predictions.csv")
    model_metrics["Regression"] = reg_metrics

    # --------------------------------------------------
    # 3️⃣ Random Forest
    # --------------------------------------------------

    rf = RandomForestModel()
    rf_model, rf_metrics, rf_predictions, rf_X_train = rf.train_or_load(df, TRAIN_MODE)

    print("Random Forest Metrics:", rf_metrics)
    save_predictions(rf_predictions, "random_forest_predictions.csv")
    model_metrics["Random Forest"] = rf_metrics

    # --------------------------------------------------
    # 4️⃣ ARIMA
    # --------------------------------------------------

    arima = ARIMAModel()
    arima_model, arima_metrics, arima_predictions = arima.train(df)

    print("ARIMA Metrics:", arima_metrics)
    save_predictions(arima_predictions, "arima_predictions.csv")
    model_metrics["ARIMA"] = arima_metrics

    # --------------------------------------------------
    # 5️⃣ SARIMA
    # --------------------------------------------------

    sarima = SARIMAModel()
    sarima_model, sarima_metrics, sarima_predictions = sarima.train(df)

    print("SARIMA Metrics:", sarima_metrics)
    save_predictions(sarima_predictions, "sarima_predictions.csv")
    model_metrics["SARIMA"] = sarima_metrics

    # --------------------------------------------------
    # 6️⃣ Anomaly Detection
    # --------------------------------------------------

    detector = ResidualAnomalyDetector()
    anomaly_df = detector.detect(reg_predictions)

    total_anomalies = anomaly_df["Anomaly"].sum()
    total_wastage = anomaly_df["Energy_Wastage"].sum()
    total_actual = anomaly_df["Actual"].sum()

    efficiency_score = 1 - (total_wastage / total_actual)

    print("\n System Insights")
    print("Total Anomalies:", total_anomalies)
    print("Total Energy Wastage:", round(total_wastage, 4))
    print("Efficiency Score:", round(efficiency_score, 4))

    # --------------------------------------------------
    # 7️⃣ Multi-Model Optimization Comparison
    # --------------------------------------------------

    print("\n Running Multi-Model Optimization Comparison...\n")

    models_for_optimization = {
        "Regression": reg_predictions,
        "Random Forest": rf_predictions,
        "ARIMA": arima_predictions
    }

    optimization_results = {}

    for model_name, predictions in models_for_optimization.items():

        possible_actual_cols = ["Actual", "Electricity_Consumed", "y_test"]
        possible_pred_cols = ["Predicted", "Prediction", "y_pred"]

        actual_col = next((c for c in possible_actual_cols if c in predictions.columns), None)
        pred_col = next((c for c in possible_pred_cols if c in predictions.columns), None)

        if actual_col is None or pred_col is None:
            print(f"Skipping {model_name} (required columns not found)")
            continue

        optimizer = EnergyOptimizer(
            predictions,
            actual_col=actual_col,
            pred_col=pred_col
        )

        current_eff = optimizer.efficiency_score()
        simulation = optimizer.simulate_load_shift(30)
        optimized_eff = simulation["optimized_efficiency"]

        gain = optimized_eff - current_eff

        optimization_results[model_name] = {
            "Current Efficiency": round(current_eff, 4),
            "Optimized Efficiency": round(optimized_eff, 4),
            "Efficiency Gain": round(gain, 4)
        }

    print("\nModel-Wise Optimization Performance:\n")

    for model, results in optimization_results.items():
        print(f"{model}:")
        print("  Current Efficiency:", results["Current Efficiency"])
        print("  Optimized Efficiency:", results["Optimized Efficiency"])
        print("  Efficiency Gain:", results["Efficiency Gain"])
        print()

    # --------------------------------------------------
    # 8️⃣ Rule-Based Recommendations
    # --------------------------------------------------

    rule_engine = EnergyOptimizationEngine()
    recommendations = rule_engine.generate_recommendations(anomaly_df)

    print("\n Optimization Recommendations:")
    for rec in recommendations:
        print("-", rec)

    # --------------------------------------------------
    # 9️⃣ Explainability Layer (Random Forest)
    # --------------------------------------------------

    print("\n Running Explainability Analysis (Random Forest)...\n")

    explainer = RandomForestExplainer(rf_model, rf_X_train)

    importance_df = explainer.global_importance()

    importance_df.to_csv(
        os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"),
        index=False
    )

    print("Top Feature Importances:")
    print(importance_df.head())

    # --------------------------------------------------
    # 🔟 Future Energy Prediction
    # --------------------------------------------------

    print("\n Future Energy Prediction Options")
    print("1 → Single Timestamp Prediction")
    print("2 → Full Day Prediction")

    choice = input("Select option (1/2): ")

    try:

        if choice == "1":

            future_date = input("Enter future date and time (YYYY-MM-DD HH:MM:SS): ")

            prediction = predict_future_consumption(future_date)

            print(f"\n Predicted Energy Consumption at {future_date}: {round(prediction,4)}")

        elif choice == "2":

            future_day = input("Enter date (YYYY-MM-DD): ")

            df_future = predict_full_day(future_day)

            print("\n Full Day Prediction Generated\n")
            print(df_future.head())

            df_future.to_csv(
                os.path.join(OUTPUT_DIR, "future_day_prediction.csv"),
                index=False
            )

            print("\n Saved to outputs/future_day_prediction.csv")

        else:

            print("Invalid option selected.")

    except Exception as e:

        print("Prediction failed:", e)

    print("\n System Execution Completed Successfully.\n")


# ======================================================
# ENTRY POINT
# ======================================================

if __name__ == "__main__":
    main()