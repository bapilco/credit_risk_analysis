from pathlib import Path

import matplotlib.pyplot as plt
import shap


class SHAPExplainer:

    def __init__(self, model, feature_names, background_data):
        self.model = model
        self.feature_names = feature_names

        masker = shap.maskers.Independent(background_data)

        self.explainer = shap.LinearExplainer(
            model.model,
            masker=masker
        )

    def explain(self, X):
        return self.explainer(X)

    def summary_plot(self, shap_values, X, output_path):
        shap.summary_plot(
            shap_values,
            X,
            feature_names=self.feature_names,
            show=False
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def waterfall_plot(self, shap_values, index, output_path):
        shap.plots.waterfall(
            shap_values[index],
            show=False
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()