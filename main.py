import os
import json
import pandas as pd
from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data
from forecasting.regression_model import RegressionModel
from forecasting.arima_model import ARIMAModel
from forecasting.random_forest_model import RandomForestModel
from forecasting.sarima_model import SARIMAModel
from anomaly_detection.residual_detector import ResidualAnomalyDetector
from optimization.rule_engine import EnergyOptimizationEngine

TRAIN_MODE = False


def main():
    print(" Starting Smart Energy Analytics System\n")


    os.makedirs("outputs", exist_ok=True)


    df = load_data()
    df = preprocess_data(df)


    regression = RegressionModel()
    reg_model, reg_metrics, reg_predictions = regression.train_or_load(df, TRAIN_MODE)
    print("Regression Metrics:", reg_metrics)

    arima = ARIMAModel()
    arima_model, arima_metrics, arima_predictions = arima.train(df)
    print("ARIMA Metrics:", arima_metrics)

    rf = RandomForestModel()
    rf_model, rf_metrics, rf_predictions = rf.train_or_load(df, TRAIN_MODE)
    print("Random Forest Metrics:", rf_metrics)

    sarima = SARIMAModel()
    sarima_model, sarima_metrics, sarima_predictions = sarima.train(df)
    print("SARIMA Metrics:", sarima_metrics)

    detector = ResidualAnomalyDetector()
    anomaly_df = detector.detect(reg_predictions)

    total_anomalies = anomaly_df["Anomaly"].sum()
    total_wastage = anomaly_df["Energy_Wastage"].sum()

    print("Total Anomalies Detected:", total_anomalies)
    print("Total Energy Wastage:", round(total_wastage, 4))

    total_actual = anomaly_df["Actual"].sum()
    efficiency_score = 1 - (total_wastage / total_actual)
    print("System Efficiency Score:", round(efficiency_score, 4))

    optimizer = EnergyOptimizationEngine()
    recommendations = optimizer.generate_recommendations(anomaly_df)

    print("\nOptimization Recommendations:")
    for rec in recommendations:
        print("-", rec)

    anomaly_df.to_csv("outputs/final_results.csv", index=False)

    metrics_dict = {
        "Regression": reg_metrics,
        "RandomForest": rf_metrics,
        "ARIMA": arima_metrics,
        "SARIMA": sarima_metrics
    }

    with open("outputs/model_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=4)

    system_summary = {
        "Total_Anomalies": int(total_anomalies),
        "Total_Wastage": float(total_wastage),
        "Efficiency_Score": float(efficiency_score)
    }

    with open("outputs/system_summary.json", "w") as f:
        json.dump(system_summary, f, indent=4)

    print("\nOutputs saved successfully in 'outputs/' folder.")
    print("\nSystem Execution Completed Successfully.")


if __name__ == "__main__":
    main()