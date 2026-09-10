from sklearn.linear_model import LogisticRegression


class LogisticModel:
    """
    Modelo base de clasificación utilizando Regresión Logística.
    """

    def __init__(self):
        self.model = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]