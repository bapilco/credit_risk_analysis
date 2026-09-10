from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)


class ModelVisualizer:

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_metrics(self, base_results, pca_results):

        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

        df = pd.DataFrame({
            "Metric": [m.capitalize() for m in metrics],
            "Baseline": [base_results[m] for m in metrics],
            "PCA": [pca_results[m] for m in metrics]
        })

        df.to_csv(
            self.output_dir / "model_comparison.csv",
            index=False
        )

    def plot_pca_variance(self, pca_model):

        variance = pca_model.cumulative_variance()

        plt.figure(figsize=(7,4))

        plt.plot(
            range(1, len(variance)+1),
            variance,
            marker="o"
        )

        plt.axhline(
            0.95,
            color="red",
            linestyle="--",
            label="95%"
        )

        plt.xlabel("Number of Components")
        plt.ylabel("Cumulative Variance")
        plt.title("PCA Explained Variance")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "pca_variance.png",
            dpi=300
        )
        plt.close()

    def plot_roc(self, base_model, pca_model, X_test, y_test):

        base_prob = base_model.predict_proba(X_test)
        pca_prob = pca_model.predict_proba(X_test)

        fpr_base, tpr_base, _ = roc_curve(y_test, base_prob)
        fpr_pca, tpr_pca, _ = roc_curve(y_test, pca_prob)

        auc_base = auc(fpr_base, tpr_base)
        auc_pca = auc(fpr_pca, tpr_pca)

        plt.figure(figsize=(6,6))

        plt.plot(
            fpr_base,
            tpr_base,
            label=f"Baseline ({auc_base:.3f})",
            linewidth=2
        )

        plt.plot(
            fpr_pca,
            tpr_pca,
            label=f"PCA ({auc_pca:.3f})",
            linewidth=2
        )

        plt.plot([0,1],[0,1],"k--", alpha=0.5)

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.grid(alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "roc_curve.png",
            dpi=300
        )
        plt.close()

    def plot_confusion_matrix(self, model, X_test, y_test):

        y_pred = model.predict(X_test)

        fig, ax = plt.subplots(figsize=(5,5))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No Mora","Mora"],
            cmap="Blues",
            colorbar=False,
            ax=ax
        )

        ax.set_title("Confusion Matrix")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "confusion_matrix.png",
            dpi=300
        )
        plt.close()

    def print_summary(self, base_results, pca_results, pca_model):

        print("\n===== MODEL COMPARISON =====\n")
        print(f"{'Metric':<12}{'Base':>10}{'PCA':>10}")

        for metric in ["accuracy","precision","recall","f1","roc_auc"]:
            print(
                f"{metric:<12}"
                f"{base_results[metric]:>10.4f}"
                f"{pca_results[metric]:>10.4f}"
            )

        print(
            f"\nPCA Components: {len(pca_model.explained_variance())}"
        )

        print(
            f"Explained Variance: "
            f"{sum(pca_model.explained_variance())*100:.2f}%"
        )