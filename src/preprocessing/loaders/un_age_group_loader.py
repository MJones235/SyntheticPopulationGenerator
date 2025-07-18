from pathlib import Path
from typing import Literal
import pandas as pd

from .base_loader import BaseLoader


class UNAgeGroupLoader(BaseLoader):
    def __init__(self, male_path: Path, female_path: Path):
        self.male_path = male_path
        self.female_path = female_path

    def load_file(self, filename: str) -> pd.DataFrame:
        male_df = self._load_sex_file(self.male_path, sex="Male")
        female_df = self._load_sex_file(self.female_path, sex="Female")
        return pd.concat([male_df, female_df], ignore_index=True)

    def _load_sex_file(self, file_path: Path, sex: Literal["Male", "Female"]) -> pd.DataFrame:
        df = pd.read_excel(file_path, skiprows=16)

        df = df[df["Type"] == "Country/Area"]
        df = df.dropna(subset=["Region, subregion, country or area *", "Year"])
        df = df.sort_values("Year").groupby("Region, subregion, country or area *").tail(1)

        age_columns = df.columns[df.columns.get_loc("0-4"):]
        df["TotalPopulation"] = df[age_columns].replace("...", pd.NA).apply(pd.to_numeric, errors="coerce").sum(axis=1)
        df = df[df["TotalPopulation"] >= 100]

        melted = df.melt(
            id_vars=["Region, subregion, country or area *"],
            value_vars=age_columns,
            var_name="Age group",
            value_name="Population"
        )
        melted["Sex"] = sex
        melted = melted.rename(columns={"Region, subregion, country or area *": "Country"})
        melted["Population"] = pd.to_numeric(melted["Population"], errors="coerce")

        return melted.dropna(subset=["Population"])
