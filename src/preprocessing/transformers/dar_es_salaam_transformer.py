from .base_transformer import BaseTransformer
import pandas as pd

class DarEsSalaamAgeTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter out "Both" to avoid double-counting
        df = df[df["Sex"].isin(["Male", "Female"])].copy()

        # Group and calculate percentages
        grouped = df.groupby(["Age group", "Sex"], as_index=False)["Population"].sum()
        total = grouped["Population"].sum()
        grouped["Percentage"] = (grouped["Population"] / total * 100).round(1)

        # Rename to final format
        return grouped.rename(columns={"Age group": "Category_1", "Sex": "Category_2"})[
            ["Category_1", "Category_2", "Percentage"]
        ]


class DarEsSalaamSexTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Filter out "Both" again to keep Male/Female only
        df = df[df["Sex"].isin(["Male", "Female"])].copy()

        # Aggregate by sex
        grouped = df.groupby("Sex", as_index=False)["Population"].sum()
        total = grouped["Population"].sum()
        grouped["Percentage"] = (grouped["Population"] / total * 100).round(1)

        return grouped.rename(columns={"Sex": "Category_1"})[
            ["Category_1", "Percentage"]
        ]
