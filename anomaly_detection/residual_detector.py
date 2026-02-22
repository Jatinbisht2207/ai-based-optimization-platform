import numpy as np
class ResidualAnomalyDetector:
    def detect(self, predictions_df, threshold=2.5):
        predictions_df["Residual"] = (
            predictions_df["Actual"] - predictions_df["Predicted"]
        )
        mean = predictions_df["Residual"].mean()
        std = predictions_df["Residual"].std()

        predictions_df["Z_score"] = (
            (predictions_df["Residual"] - mean) / std
        )
        predictions_df["Anomaly"] = (
            np.abs(predictions_df["Z_score"]) > threshold
        )
        predictions_df["Energy_Wastage"] = np.where(
            predictions_df["Residual"] > 0,
            predictions_df["Residual"],
            0
        )

        return predictions_df