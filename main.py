import os
import json
import pandas as pd
from ingestion.data_loader import load_data
from preprocessing.preprocess import preprocess_data
from forecasting.regression_model import RegressionModel
from forecasting.random_forest_model import RandomForestModel
from forecasting.arima_model import ARIMAModel
from forecasting.sarima_model import SARIMAModel
from anomaly_detection.residual_detector import ResidualAnomalyDetector
from optimization.rule_engine import EnergyOptimizationEngine


TRAIN_MODE = False
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)



def save_predictions(df, filename):
    """
    Save full prediction dataframe including Timestamp
    """
    df.to_csv(
        os.path.join(OUTPUT_DIR, filename),
        index=False
    )

def main():

    print("\n🚀 Starting Smart Energy Analytics System\n")

    # 1️⃣ Load + Preprocess
    df = load_data()
    df = preprocess_data(df)

    model_metrics = {}

    regression = RegressionModel()
    reg_model, reg_metrics, reg_predictions = regression.train_or_load(df, TRAIN_MODE)

    print("Regression Metrics:", reg_metrics)

    save_predictions(reg_predictions, "regression_predictions.csv")
    model_metrics["Regression"] = reg_metrics


    rf = RandomForestModel()
    rf_model, rf_metrics, rf_predictions = rf.train_or_load(df, TRAIN_MODE)

    print("Random Forest Metrics:", rf_metrics)

    save_predictions(rf_predictions, "random_forest_predictions.csv")
    model_metrics["Random Forest"] = rf_metrics

   
    arima = ARIMAModel()
    arima_model, arima_metrics, arima_predictions = arima.train(df)

    print("ARIMA Metrics:", arima_metrics)

    save_predictions(arima_predictions, "arima_predictions.csv")
    model_metrics["ARIMA"] = arima_metrics

    # =====================================================
# 5️⃣ SARIMA MODEL
# =====================================================
    sarima = SARIMAModel()
    sarima_model, sarima_metrics, sarima_predictions = sarima.train(df)

    print("SARIMA Metrics:", sarima_metrics)

    save_predictions(sarima_predictions, "sarima_predictions.csv")
    model_metrics["SARIMA"] = sarima_metrics

    detector = ResidualAnomalyDetector()
    anomaly_df = detector.detect(reg_predictions)

    total_anomalies = anomaly_df["Anomaly"].sum()
    total_wastage = anomaly_df["Energy_Wastage"].sum()
    total_actual = anomaly_df["Actual"].sum()

    efficiency_score = 1 - (total_wastage / total_actual)

    print("\n📊 System Insights")
    print("Total Anomalies:", total_anomalies)
    print("Total Energy Wastage:", round(total_wastage, 4))
    print("Efficiency Score:", round(efficiency_score, 4))

    anomaly_df.to_csv(os.path.join(OUTPUT_DIR, "final_results.csv"), index=False)

   
    system_summary = {
        "Total_Anomalies": int(total_anomalies),
        "Total_Wastage": float(total_wastage),
        "Efficiency_Score": float(efficiency_score)
    }

    with open(os.path.join(OUTPUT_DIR, "system_summary.json"), "w") as f:
        json.dump(system_summary, f, indent=4)

    with open(os.path.join(OUTPUT_DIR, "model_metrics.json"), "w") as f:
        json.dump(model_metrics, f, indent=4)

    optimizer = EnergyOptimizationEngine()
    recommendations = optimizer.generate_recommendations(anomaly_df)

    print("\n🔧 Optimization Recommendations:")
    for rec in recommendations:
        print("-", rec)

    print("\nSystem Execution Completed Successfully.\n")

if __name__ == "__main__":
    main()