from sklearn.model_selection import StratifiedKFold, cross_validate
import pandas as pd
from sklearn.model_selection import cross_val_predict


class CrossValidator:

    @staticmethod
    def evaluate(model, X, y, folds=5):

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc"
        }

        scores = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        results = pd.DataFrame({
            "accuracy": scores["test_accuracy"],
            "precision": scores["test_precision"],
            "recall": scores["test_recall"],
            "f1": scores["test_f1"],
            "roc_auc": scores["test_roc_auc"]
        })

        return results



    @staticmethod
    def predict(model, X, y, folds=5):
        """
        Genera predicciones y probabilidades mediante
        validación cruzada estratificada.
        """

        cv = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=42
        )

        # Predicción de clases (0 o 1)
        y_pred = cross_val_predict(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            n_jobs=-1
        )

        # Probabilidades de la clase positiva
        y_prob = cross_val_predict(
            estimator=model,
            X=X,
            y=y,
            cv=cv,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        return y, y_pred, y_prob