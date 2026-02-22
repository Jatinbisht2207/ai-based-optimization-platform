class EnergyOptimizationEngine:

    def generate_recommendations(self, anomaly_df):
        recommendations = []
        anomaly_count = anomaly_df["Anomaly"].sum()
        total_wastage = anomaly_df["Energy_Wastage"].sum()
        if anomaly_count > 5:
            recommendations.append(
                "Frequent anomalies detected. Consider system inspection."
            )
        if total_wastage > 50:
            recommendations.append(
                "High cumulative energy wastage observed. Implement load optimization."
            )
        # Example: Check night hour inefficiency
        if "hour" in anomaly_df.columns:
            night_wastage = anomaly_df[
                (anomaly_df["hour"] < 6) &
                (anomaly_df["Energy_Wastage"] > 0)
            ]
            if len(night_wastage) > 0:
                recommendations.append(
                    "Off-hour consumption detected. Consider shutting down idle equipment."
                )
        if not recommendations:
            recommendations.append("System operating within acceptable limits.")
        return recommendations