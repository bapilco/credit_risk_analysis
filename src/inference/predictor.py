import pandas as pd


class CreditPredictor:

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, df):

        df = df.copy()

        # Tomar la primera columna como identificador
        id_column = df.columns[0]

        ids = df[id_column]

        X = df.drop(columns=[id_column])

        X = X.drop(columns=["SeriousDlqin2yrs"])

        X = X.rename(columns={
            "age": "Age"
        })

        X = self.preprocessor.transform(X)

        probability = self.model.predict_proba(X)
        prediction = (probability >= 0.5).astype(int)

        return pd.DataFrame({
            "Id": ids,
            "Probability_Default": probability.round(4),
            "Prediction": prediction
        })