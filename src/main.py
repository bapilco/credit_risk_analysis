from pathlib import Path

import pandas as pd
import shap

from data.loader import DataLoader
from data.preprocessing import Preprocessor

from models.logistic_model import LogisticModel
from models.pca_model import PCAModel

from evaluation.cross_validator import CrossValidator
from evaluation.metrics import MetricsEvaluator
from evaluation.visualizer import ModelVisualizer

from explainability.shap_explainer import SHAPExplainer
from inference.predictor import CreditPredictor

RESULTS_DIR = Path("results")


def generate_shap(model, feature_names, X):

    X_df = pd.DataFrame(X, columns=feature_names)

    background = shap.sample(
        X_df,
        100,
        random_state=42
    )

    explainer = SHAPExplainer(
        model,
        feature_names,
        background
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
    # 1. CARGA DEL DATASET
    # =====================================================

    df = DataLoader(
        "data/processed/training_clean.parquet"
    ).load()

    preprocessor = Preprocessor()

    X, y = preprocessor.process(df)

    feature_names = df.drop(
        columns=["SeriousDlqin2yrs"]
    ).columns

    # =====================================================
    # 2. MODELO BASE
    # =====================================================

    baseline_model = LogisticModel()

    # =====================================================
    # 3. 5-FOLD CROSS VALIDATION
    # =====================================================

    cv_results = CrossValidator.evaluate(
        baseline_model.model,
        X,
        y
    )

    metrics = cv_results.mean().to_dict()

    y_true, y_pred, y_prob = CrossValidator.predict(
        baseline_model.model,
        X,
        y
    )

    MetricsEvaluator.evaluate(
        y_true,
        y_pred,
        y_prob
    )

    # =====================================================
    # 4. ENTRENAR MODELO FINAL PCA
    # =====================================================

    pca_model = PCAModel()
    pca_model.train(X, y)

    # =====================================================
    # 5. VISUALIZACIONES
    # =====================================================

    visualizer = ModelVisualizer()

    visualizer.save_metrics(metrics)

    visualizer.print_summary(
        metrics,
        pca_model
    )

    visualizer.plot_confusion_matrix(
        y_true,
        y_pred
    )

    visualizer.plot_roc(
        y_true,
        y_prob
    )

    visualizer.plot_scree(X)

    visualizer.plot_pca_variance(
        pca_model
    )

    # =====================================================
    # 6. ENTRENAR MODELO FINAL BASE (PARA SHAP)
    # =====================================================
    
    baseline_model.train(X, y)
    
    generate_shap(
        baseline_model,
        feature_names,
        X
    )

    # =====================================================
    # 6. SHAP (MODELO FINAL)
    # =====================================================

    generate_shap(
        baseline_model,
        feature_names,
        X
    )

    # =====================================================
    # 7. INFERENCIA
    # =====================================================

    generate_predictions(
        pca_model,
        preprocessor
    )


if __name__ == "__main__":
    main()