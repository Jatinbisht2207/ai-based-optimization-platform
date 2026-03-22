import pandas as pd
import numpy as np


class EnergyOptimizer:

    def __init__(self, df, actual_col, pred_col):
        self.df = df.copy()
        self.actual_col = actual_col
        self.pred_col = pred_col

        self.df["Wastage"] = np.where(
            self.df[self.actual_col] > self.df[self.pred_col],
            self.df[self.actual_col] - self.df[self.pred_col],
            0
        )

        self.df["Hour"] = pd.to_datetime(self.df["Timestamp"]).dt.hour

    # --------------------------------------------------
    # Basic Metrics
    # --------------------------------------------------

    def total_wastage(self):
        return self.df["Wastage"].sum()

    def wastage_percentage(self):
        total_consumption = self.df[self.actual_col].sum()
        return (self.total_wastage() / total_consumption) * 100

    def efficiency_score(self):
        return 100 - self.wastage_percentage()

    # --------------------------------------------------
    # Peak & Off-Peak Analysis
    # --------------------------------------------------

    def peak_hours(self, top_n=3):
        hourly_wastage = self.df.groupby("Hour")["Wastage"].sum()
        return hourly_wastage.sort_values(ascending=False).head(top_n)

    def lowest_load_hours(self, bottom_n=3):
        hourly_consumption = self.df.groupby("Hour")[self.actual_col].sum()
        return hourly_consumption.sort_values().head(bottom_n)

    # --------------------------------------------------
    # Load Shifting Simulation
    # --------------------------------------------------

    def simulate_load_shift(self, shift_percent=30):
        """
        Simulate shifting % of peak-hour wastage to lowest load hours
        """

        shift_factor = shift_percent / 100

        peak_hours = self.peak_hours()
        lowest_hours = self.lowest_load_hours()

        peak_wastage_total = peak_hours.sum()

        # Assume we can reduce peak wastage proportionally
        reduced_wastage = peak_wastage_total * shift_factor

        optimized_wastage = self.total_wastage() - reduced_wastage

        total_consumption = self.df[self.actual_col].sum()
        optimized_efficiency = 100 - (
            (optimized_wastage / total_consumption) * 100
        )

        return {
            "shift_percent": shift_percent,
            "peak_hours": list(peak_hours.index),
            "lowest_hours": list(lowest_hours.index),
            "reduced_wastage": reduced_wastage,
            "optimized_wastage": optimized_wastage,
            "optimized_efficiency": optimized_efficiency
        }