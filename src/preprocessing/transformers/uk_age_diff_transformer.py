import pandas as pd
import re
from .base_transformer import BaseTransformer

class UKPartnerAgeDiffTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        def extract_age_diff(label: str) -> int:
            match = re.search(r"(\d+)", label)
            sign = -1 if "Female older" in label else 1
            return sign * int(match.group(1)) if match else 0

        df = df[["Age disparity", "Opposite-sex married couples, 2021"]].dropna()
        df["Category_1"] = df["Age disparity"].apply(extract_age_diff)

        total = df["Opposite-sex married couples, 2021"].sum()
        df["Percentage"] = (df["Opposite-sex married couples, 2021"] / total * 100).round(1)

        return df[["Category_1", "Percentage"]].sort_values("Category_1")