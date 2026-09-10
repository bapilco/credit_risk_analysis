from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class PCAModel:
    """
    Modelo de Regresión Logística utilizando
    reducción de dimensionalidad mediante PCA.
    """

    def __init__(self, n_components=0.95):

        self.pca = PCA(n_components=n_components)

        self.model = LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )

    def train(self, X_train, y_train):

        X_train_pca = self.pca.fit_transform(X_train)

        self.model.fit(X_train_pca, y_train)

    def predict(self, X_test):

        X_test_pca = self.pca.transform(X_test)

        return self.model.predict(X_test_pca)

    def predict_proba(self, X_test):

        X_test_pca = self.pca.transform(X_test)

        return self.model.predict_proba(X_test_pca)[:, 1]

    def explained_variance(self):

        return self.pca.explained_variance_ratio_

    def cumulative_variance(self):
        return self.pca.explained_variance_ratio_.cumsum()