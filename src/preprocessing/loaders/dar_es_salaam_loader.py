from .base_loader import BaseLoader
import pandas as pd
from pathlib import Path
import re

class DarEsSalaamLoader(BaseLoader):
    def __init__(self, filepath: Path):
        self.filepath = filepath

    def load_file(self, filename: str) -> pd.DataFrame:
        if filename in {"age_group.csv", "sex.csv"}:
            return self._load_combined_data()
        else:
            raise ValueError(f"Unknown filename: {filename}")

    def _load_combined_data(self) -> pd.DataFrame:
        df = pd.read_excel(self.filepath, sheet_name="2015 - 2030", skiprows=3)
        df = df[df["NSO_NAME"] == "Dar es Salaam"]

        # Keep only relevant columns
        pattern = re.compile(r"^[BMF](\d{4}|80PL)_2025$")
        population_cols = [col for col in df.columns if pattern.match(col)]

        # Melt the population columns
        melted = df[population_cols].melt(var_name="colname", value_name="Population")

        # Extract sex, age, year from colname
        def parse_col(col):
            match = re.match(r"^([BMF])(\d{4}|80PL)_2025$", col)
            if not match:
                return None
            sex_code, age_code = match.groups()
            sex = {"M": "Male", "F": "Female", "B": "Both"}[sex_code]
            if age_code == "80PL":
                age_group = "80+"
            else:
                age_group = f"{int(age_code[:2])}-{int(age_code[2:])}"
            return pd.Series({"Sex": sex, "Age group": age_group})

        melted = melted.join(melted["colname"].apply(parse_col))
        melted = melted.drop(columns=["colname"])
        melted = melted.dropna(subset=["Sex", "Age group"])

        return melted
