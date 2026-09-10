from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.income_median = None
        self.dependents_median = None

    def process(self, df):

        X = df.drop(columns=["SeriousDlqin2yrs"])
        y = df["SeriousDlqin2yrs"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        # 1. Aprender medianas SOLO del entrenamiento
        self.income_median = X_train["MonthlyIncome"].median()
        self.dependents_median = X_train["NumberOfDependents"].median()

        # 2. Imputar entrenamiento
        X_train["MonthlyIncome"] = X_train["MonthlyIncome"].fillna(
            self.income_median
        )
        X_train["NumberOfDependents"] = X_train[
            "NumberOfDependents"
        ].fillna(self.dependents_median)

        # 3. Imputar prueba usando las mismas medianas
        X_test["MonthlyIncome"] = X_test["MonthlyIncome"].fillna(
            self.income_median
        )
        X_test["NumberOfDependents"] = X_test[
            "NumberOfDependents"
        ].fillna(self.dependents_median)

        # 4. Escalado
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test

    def transform(self, X):
        X = X.copy()

        X["MonthlyIncome"] = X["MonthlyIncome"].fillna(
            self.income_median
        )
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(
            self.dependents_median
        )

        return self.scaler.transform(X)