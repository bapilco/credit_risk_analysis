from pathlib import Path
import pandas as pd


class DataLoader:

    def __init__(self, data_path):
        self.data_path = Path(data_path)

    def load(self):

        suffix = self.data_path.suffix.lower()

        if suffix == ".parquet":
            return pd.read_parquet(self.data_path)

        if suffix == ".csv":
            return pd.read_csv(self.data_path)

        raise ValueError(f"Formato no soportado: {suffix}")