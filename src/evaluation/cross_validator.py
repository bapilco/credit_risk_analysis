import pandas as pd

from sklearn.model_selection import StratifiedKFold

from evaluation.metrics import MetricsEvaluator


class CrossValidator:

    @staticmethod
    def evaluate(model, X, y, n_splits=5):

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        results = []

        for train_idx, test_idx in skf.split(X, y):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.train(X_train, y_train)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)

            metrics = MetricsEvaluator.evaluate(
                y_test,
                y_pred,
                y_prob
            )

            results.append(metrics)

        return pd.DataFrame(results)

    @staticmethod
    def predict(model, X, y, n_splits=5):

        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=42
        )

        y_true = []
        y_pred = []
        y_prob = []

        for train_idx, test_idx in skf.split(X, y):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.train(X_train, y_train)

            y_true.extend(y_test)
            y_pred.extend(model.predict(X_test))
            y_prob.extend(model.predict_proba(X_test))

        return y_true, y_pred, y_prob