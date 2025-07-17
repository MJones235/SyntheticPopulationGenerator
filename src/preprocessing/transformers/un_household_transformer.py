from .base_transformer import BaseTransformer
import pandas as pd

class UNHouseholdTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[["Category_1", "Percentage"]].copy()
