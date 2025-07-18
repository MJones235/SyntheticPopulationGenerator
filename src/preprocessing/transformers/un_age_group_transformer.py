from .base_transformer import BaseTransformer
import pandas as pd


class UNAgeGroupTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects a DataFrame with columns: Age group, Sex, Population
        Outputs a DataFrame with Category_1 (age), Category_2 (sex), Percentage
        """

        grouped = df.groupby(["Age group", "Sex"], as_index=False)["Population"].sum()

        total = grouped["Population"].sum()
        grouped["Percentage"] = (grouped["Population"] / total * 100).round(1)

        return grouped.rename(columns={
            "Age group": "Category_1",
            "Sex": "Category_2"
        })[["Category_1", "Category_2", "Percentage"]]
