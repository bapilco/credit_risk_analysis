from pathlib import Path

import pandas as pd
import shap

from data.loader import DataLoader
from data.preprocessing import Preprocessor

from models.logistic_model import LogisticModel
from models.pca_model import PCAModel

from evaluation.metrics import MetricsEvaluator
from evaluation.visualizer import ModelVisualizer

from explainability.shap_explainer import SHAPExplainer
from inference.predictor import CreditPredictor


RESULTS_DIR = Path("results")


def train_model(model, X_train, X_test, y_train, y_test):
    """Entrena un modelo y retorna sus métricas."""

    model.train(X_train, y_train)

    return MetricsEvaluator.evaluate(
        y_test,
        model.predict(X_test),
        model.predict_proba(X_test)
    )


def generate_shap(baseline_model, feature_names, X_train, X_test):
    """Genera las visualizaciones SHAP."""

    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    background = shap.sample(
        X_train_df,
        100,
        random_state=42
    )

    explainer = SHAPExplainer(
        baseline_model,
        feature_names,
        background
    )

    shap_values = explainer.explain(X_test_df)

    explainer.summary_plot(
        shap_values,
        X_test_df,
        RESULTS_DIR / "shap_summary.png"
    )

    explainer.waterfall_plot(
        shap_values,
        index=0,
        output_path=RESULTS_DIR / "shap_waterfall.png"
    )


def generate_predictions(pca_model, preprocessor):
    """
    Realiza inferencia sobre clientes nuevos
    utilizando cs-test.csv.
    """

    test_df = DataLoader("data/raw/cs-test.csv").load()

    predictor = CreditPredictor(
        model=pca_model,
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
    # 1. CARGA Y PREPROCESAMIENTO (TRAINING)
    # =====================================================

    training_df = DataLoader(
        "data/processed/training_clean.parquet"
    ).load()

    preprocessor = Preprocessor()

    X_train, X_test, y_train, y_test = preprocessor.process(training_df)

    # =====================================================
    # 2. ENTRENAMIENTO DE MODELOS
    # =====================================================

    baseline_model = LogisticModel()
    pca_model = PCAModel()

    baseline_results = train_model(
        baseline_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    pca_results = train_model(
        pca_model,
        X_train,
        X_test,
        y_train,
        y_test
    )

    # =====================================================
    # 3. VISUALIZACIONES Y MÉTRICAS
    # =====================================================

    visualizer = ModelVisualizer()

    visualizer.save_metrics(
        baseline_results,
        pca_results
    )

    visualizer.plot_pca_variance(pca_model)

    visualizer.plot_roc(
        baseline_model,
        pca_model,
        X_test,
        y_test
    )

    visualizer.plot_confusion_matrix(
        pca_model,
        X_test,
        y_test
    )

    visualizer.print_summary(
        baseline_results,
        pca_results,
        pca_model
    )

    # =====================================================
    # 4. SHAP
    # =====================================================

    feature_names = training_df.drop(
        columns=["SeriousDlqin2yrs"]
    ).columns

    generate_shap(
        baseline_model,
        feature_names,
        X_train,
        X_test
    )

    # =====================================================
    # 5. INFERENCIA SOBRE CS-TEST
    # =====================================================

    generate_predictions(
        pca_model,
        preprocessor
    )


if __name__ == "__main__":
    main()