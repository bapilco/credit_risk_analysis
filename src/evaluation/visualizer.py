from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


class ModelVisualizer:

    def __init__(self, output_dir="results"):

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    # =====================================================
    # MÉTRICAS
    # =====================================================

    def save_metrics(self, metrics):

        pd.DataFrame([metrics]).round(4).to_csv(
            self.output_dir / "model_metrics.csv",
            index=False
        )

        print("Métricas guardadas en results/model_metrics.csv")

#    def print_summary(self, metrics, pca_model):
#
#        print("\n===== RESULTADOS CROSS VALIDATION =====\n")
#
#        print(f"{'Métrica':<12} {'Valor':<10}")
#
#        for metric in [
#            "accuracy",
#            "precision",
#            "recall",
#            "f1",
#            "roc_auc"
#        ]:
#            print(f"{metric:<12} {metrics[metric]:<10.4f}")
#
#        print("\nComponentes retenidos:",
#              len(pca_model.explained_variance()))
#
#        print(
#            "Varianza explicada:",
#            f"{sum(pca_model.explained_variance()) * 100:.2f}%"
#        )

    # =====================================================
    # PCA
    # =====================================================

    def plot_pca_variance(self, pca_model):

        variance = pca_model.cumulative_variance()

        plt.figure(figsize=(7, 4))

        plt.plot(
            range(1, len(variance) + 1),
            variance,
            marker="o",
            linewidth=2
        )

        plt.axhline(
            y=0.95,
            color="red",
            linestyle="--",
            label="95%"
        )

        plt.xticks(range(1, len(variance) + 1))
        plt.xlabel("Número de componentes")
        plt.ylabel("Varianza acumulada")
        plt.title("Varianza explicada por PCA")

        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "pca_variance.png",
            dpi=300
        )
        plt.close()

        print("Gráfico guardado -> results/pca_variance.png")

    def plot_scree(self, X):

        pca = PCA()
        pca.fit(X)

        explained = pca.explained_variance_ratio_

        plt.figure(figsize=(7, 4))

        plt.plot(
            range(1, len(explained) + 1),
            explained,
            marker="o",
            linewidth=2
        )

        plt.axvline(
            x=4,
            color="red",
            linestyle="--",
            alpha=0.8,
            label="Codo visual"
        )

        plt.xticks(range(1, len(explained) + 1))
        plt.xlabel("Componentes principales")
        plt.ylabel("Varianza explicada")
        plt.title("Scree Plot - PCA")

        plt.grid(alpha=0.3)
        plt.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "scree_plot.png",
            dpi=300
        )
        plt.close()

        print("Gráfico guardado -> results/scree_plot.png")

    # =====================================================
    # CROSS VALIDATION
    # =====================================================
    def plot_roc(self, y_true, y_prob, model_name):

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 5))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.3f}"
        )

        plt.plot([0, 1], [0, 1], "--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {model_name.upper()}")

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"results/roc_{model_name}.png",
            dpi=300
        )

        plt.close()

        print(f"Gráfico guardado -> results/roc_{model_name}.png")

    def plot_confusion_matrix(self, y_true, y_pred, model_name):

        fig, ax = plt.subplots(figsize=(5, 5))

        ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            cmap="Blues",
            ax=ax
        )

        ax.set_title(f"Confusion Matrix - {model_name.upper()}")

        plt.tight_layout()

        plt.savefig(
            f"results/confusion_{model_name}.png",
            dpi=300
        )

        plt.close()

        print(f"Gráfico guardado -> results/confusion_{model_name}.png")