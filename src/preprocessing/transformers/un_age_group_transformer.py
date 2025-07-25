import re
from .base_transformer import BaseTransformer
import pandas as pd


class UNAgeGroupTransformer(BaseTransformer):
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Collapse all age groups 85+ into "85+"
        def normalize_age_group(label):
            match = re.match(r"(\d+)", label)
            if match and int(match.group(1)) >= 85:
                return "85+"
            return label

        df = df.copy()
        df["Age group"] = df["Age group"].apply(normalize_age_group)

        grouped = df.groupby(["Age group", "Sex"], as_index=False)["Population"].sum()

        total = grouped["Population"].sum()
        grouped["Percentage"] = (grouped["Population"] / total * 100).round(1)

        return grouped.rename(columns={
            "Age group": "Category_1",
            "Sex": "Category_2"
        })[["Category_1", "Category_2", "Percentage"]]


    def extract_sex_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expects columns: Age group, Sex, Population
        Outputs: Category_1 (Sex), Percentage
        """
        sex_totals = df.groupby("Sex", as_index=False)["Population"].sum()
        total = sex_totals["Population"].sum()
        sex_totals["Percentage"] = (sex_totals["Population"] / total * 100).round(1)
        return sex_totals.rename(columns={"Sex": "Category_1"})[["Category_1", "Percentage"]]
