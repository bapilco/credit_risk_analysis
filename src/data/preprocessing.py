from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessor:

    def __init__(self):
        self.scaler = StandardScaler()
        self.income_median = None
        self.dependents_median = None

    def process(self, df):
    
        # Variables predictoras y objetivo
        X = df.drop(columns=["SeriousDlqin2yrs"]).copy()
        y = df["SeriousDlqin2yrs"].copy()
    
        # Aprender las medianas
        self.income_median = X["MonthlyIncome"].median()
        self.dependents_median = X["NumberOfDependents"].median()
    
        # Imputación
        X["MonthlyIncome"] = X["MonthlyIncome"].fillna(
            self.income_median
        )
    
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(
            self.dependents_median
        )
    
        # Guardar el orden de las columnas
        self.feature_names = X.columns.tolist()
    
        # Escalado
        X = self.scaler.fit_transform(X)
    
        return X, y

    def transform(self, X):
        X = X.copy()

        X["MonthlyIncome"] = X["MonthlyIncome"].fillna(
            self.income_median
        )
        X["NumberOfDependents"] = X["NumberOfDependents"].fillna(
            self.dependents_median
        )

        return self.scaler.transform(X)