from pathlib import Path

import pandas as pd

from data.loader import DataLoader
from data.preprocessing import Preprocessor

from models.logistic_model import LogisticModel
from models.pca_model import PCAModel

from evaluation.cross_validator import CrossValidator
from evaluation.visualizer import ModelVisualizer

from explainability.shap_explainer import SHAPExplainer
from inference.predictor import CreditPredictor

RESULTS_DIR = Path("results")

def generate_shap(model, feature_names, X):

    # Recuperar nombres de las columnas
    X_df = pd.DataFrame(X, columns=feature_names)

    explainer = SHAPExplainer(
        model,
        feature_names,
        X_df
    )

    shap_values = explainer.explain(X_df)

    explainer.summary_plot(
        shap_values,
        X_df,
        RESULTS_DIR / "shap_summary.png"
    )

    explainer.waterfall_plot(
        shap_values,
        index=0,
        output_path=RESULTS_DIR / "shap_waterfall.png"
    )

    print("Gráficos SHAP guardados en results/shap_summary.png y results/shap_waterfall.png")

def generate_predictions(model, preprocessor):

    test_df = DataLoader(
        "data/raw/cs-test.csv"
    ).load()

    predictor = CreditPredictor(
        model=model,
        preprocessor=preprocessor
    )

    predictions = predictor.predict(test_df)

    predictions.to_csv(
        RESULTS_DIR / "credit_predictions.csv",
        index=False
    )

    print("Predicciones guardadas en results/credit_predictions.csv")


def main():

    RESULTS_DIR.mkdir(exist_ok=True)

    # =====================================================
    # 1. CARGA Y PREPROCESAMIENTO
    # =====================================================

    df = DataLoader(
        "data/processed/training_clean.parquet"
    ).load()

    preprocessor = Preprocessor()

    X, y = preprocessor.process(df)

    feature_names = preprocessor.feature_names

    visualizer = ModelVisualizer()

    # =====================================================
    # 2. BASELINE + CROSS VALIDATION
    # =====================================================

    baseline_model = LogisticModel()

    baseline_cv = CrossValidator.evaluate(
        baseline_model,
        X,
        y
    )

    baseline_metrics = baseline_cv.mean(numeric_only=True).to_dict()

    y_true_base, y_pred_base, y_prob_base = CrossValidator.predict(
        baseline_model,
        X,
        y
    )

    # =====================================================
    # 3. PCA + CROSS VALIDATION
    # =====================================================

    pca_model = PCAModel()

    pca_cv = CrossValidator.evaluate(
        pca_model,
        X,
        y
    )

    pca_metrics = pca_cv.mean(numeric_only=True).to_dict()

    y_true_pca, y_pred_pca, y_prob_pca = CrossValidator.predict(
        pca_model,
        X,
        y
    )

    # =====================================================
    # 4. COMPARACIÓN DE MODELOS
    # =====================================================

    comparison = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC"
        ],
        "Baseline": [
            baseline_metrics["accuracy"],
            baseline_metrics["precision"],
            baseline_metrics["recall"],
            baseline_metrics["f1"],
            baseline_metrics["roc_auc"]
        ],
        "PCA": [
            pca_metrics["accuracy"],
            pca_metrics["precision"],
            pca_metrics["recall"],
            pca_metrics["f1"],
            pca_metrics["roc_auc"]
        ]
    })

    comparison.to_csv(
        RESULTS_DIR / "model_comparison.csv",
        index=False
    )

    print("\n===== COMPARACIÓN CROSS VALIDATION =====\n")
    print(comparison.round(4))

    # =====================================================
    # 5. ENTRENAR MODELOS FINALES (100%)
    # =====================================================

    baseline_model.train(X, y)
    pca_model.train(X, y)

    # =====================================================
    # 6. VISUALIZACIONES BASELINE
    # =====================================================

    print("\n===== PCA =====")
    print(f"Componentes retenidos: {pca_model.pca.n_components_}")
    print(
        f"Varianza explicada: "
        f"{pca_model.cumulative_variance()[-1] * 100:.2f}%"
    )

    visualizer.plot_confusion_matrix(
        y_true_base,
        y_pred_base,
        "baseline"
    )

    visualizer.plot_roc(
        y_true_base,
        y_prob_base,
        "baseline"
    )

    # =====================================================
    # 7. VISUALIZACIONES PCA
    # =====================================================

    visualizer.plot_confusion_matrix(
        y_true_pca,
        y_pred_pca,
        "pca"
    )

    visualizer.plot_roc(
        y_true_pca,
        y_prob_pca,
        "pca"
    )

    visualizer.plot_scree(X)

    visualizer.plot_pca_variance(pca_model)

    # =====================================================
    # 8. SHAP (BASELINE)
    # =====================================================

    generate_shap(
        baseline_model,
        feature_names,
        X
    )

    # =====================================================
    # 9. INFERENCIA (PCA)
    # =====================================================

    generate_predictions(
        pca_model,
        preprocessor
    )


if __name__ == "__main__":
    main()