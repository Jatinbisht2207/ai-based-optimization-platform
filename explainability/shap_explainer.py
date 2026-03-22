import shap
import pandas as pd
import numpy as np


class RandomForestExplainer:

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = shap.TreeExplainer(model)

    # --------------------------------------------------
    # Global Feature Importance (Mean SHAP)
    # --------------------------------------------------

    def global_importance(self):
        shap_values = self.explainer.shap_values(self.X_train)
        mean_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "Feature": self.X_train.columns,
            "Importance": mean_shap
        }).sort_values(by="Importance", ascending=False)

        return importance_df

    # --------------------------------------------------
    # Local Explanation for Single Instance
    # --------------------------------------------------

    def local_explanation(self, index):
        shap_values = self.explainer.shap_values(self.X_train)
        return shap_values[index]