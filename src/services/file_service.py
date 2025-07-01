import os
import json
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import re

class FileService:
    PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/schemas/")
    CENSUS_DATA = os.path.join(os.path.dirname(__file__), "../../data/aggregate/")
    MICRODATA = os.path.join(os.path.dirname(__file__), "../../data/microdata/individual_uk.csv")

    def load_prompt(self, filename: str, replacements: dict = None) -> str:
        """Loads a prompt from file and applies replacements."""
        filepath = os.path.join(self.PROMPT_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as file:
            prompt = file.read()

        if replacements:
            for key, value in replacements.items():
                prompt = prompt.replace(f"{{{key}}}", str(value))

        return prompt

    def load_schema(self, filename: str):
        """Loads the household validation schema from file."""
        filepath = os.path.join(self.SCHEMA_PATH, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)

    def load_html_report(self, report_path):
        """Reads and returns the HTML content of a report."""
        try:
            with open(report_path, "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return None

    def load_household_size(self, location: str) -> Dict:
        return self._load_distribution(
            location, "household_size.csv",
            key_col="Household size (9 categories) Code",
            value_col="Observation",
            drop_keys=[0]
        )

        
    def load_household_composition(self, location: str) -> Dict:
        return self._load_distribution(
            location, "household_composition.csv",
            key_col="Household composition (8 categories)",
            value_col="Observation",
            return_type="dataframe",
            rename_key="Household Composition"
        )


    def load_age_pyramid(self, location: str) -> pd.DataFrame:
        try:
            df = self._load_csv(location, "age_group.csv")

            # Clean column names
            df.columns = df.columns.str.strip()
            df["Percentage"] = df["Observation"] / df["Observation"].sum() * 100
            df["Percentage"] = df["Percentage"].round(1)

            # Clean up age group labels
            df["Age (B) (18 categories)"] = (
                df["Age (B) (18 categories)"]
                .str.replace(r"Aged (\d+) years and under", r"0–\1", regex=True)
                .str.replace(r"Aged (\d+) to (\d+) years", r"\1–\2", regex=True)
                .str.replace(r"Aged (\d+) years and over", r"\1+", regex=True)
                .str.strip()
            )

            # Pivot into age_group × sex → percentage
            pyramid_df = df.pivot_table(index="Age (B) (18 categories)", columns="Sex (2 categories)", values="Percentage", aggfunc="sum").fillna(0)

            # Standardize sex column names
            pyramid_df.columns = [c.strip().capitalize() for c in pyramid_df.columns]
            pyramid_df = pyramid_df.reindex(columns=["Male", "Female"], fill_value=0)

            # Sort by age group
            def sort_key(label):
                import re
                match = re.match(r"(\d+)", label)
                return int(match.group(1)) if match else float("inf")

            pyramid_df = pyramid_df.sort_index(key=lambda x: x.map(sort_key))
            pyramid_df = pyramid_df.reset_index().rename(columns={"Age (B) (18 categories)": "age_group"})

            return pyramid_df

        except Exception as e:
            print(f"Failed to load age pyramid for {location}: {e}")
            return pd.DataFrame(columns=["Male", "Female"])

    def load_occupation_distribution(self, location: str) -> dict:
        return self._load_distribution(
            location, "occupation.csv",
            key_col="Occupation (current) (10 categories) Code",
            value_col="Observation"
        )

    def generate_unique_filename(self, directory: str, base_filename: str) -> str:
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, base_filename)
        filename, extension = os.path.splitext(base_filename)

        counter = 1
        while os.path.exists(filepath):
            new_filename = f"{filename}_{counter}{extension}"
            filepath = os.path.join(directory, new_filename)
            counter += 1

        return filepath
    
    def load_microdata(self, region: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.MICRODATA)
            df = df[df["region"] == region]
            return df
        except Exception as e:
            print(f"Error loading microdata for {region}: {e}")
            return pd.DataFrame()

    def _load_distribution(
        self,
        location: str,
        filename: str,
        key_col: str,
        value_col: str,
        return_type: str = "dict",
        drop_keys: Optional[List[Any]] = None,
        rename_key: Optional[str] = None,
    ) -> Union[Dict, pd.DataFrame]:
        try:
            df = self._load_csv(location, filename)
            df = self._validate_columns(df, [key_col, value_col])

            if rename_key:
                df[rename_key] = df[key_col]
                key_col = rename_key

            total = df[value_col].sum()
            df["Value"] = df[value_col] / total * 100
            df["Value"] = df["Value"].round(1)

            if drop_keys:
                df = df[~df[key_col].isin(drop_keys)]

            if return_type == "dict":
                return dict(zip(df[key_col], df["Value"]))
            else:
                return df[[key_col, "Value"]]

        except Exception as e:
            print(f"Error processing {filename} for {location}: {e}")
            return {} if return_type == "dict" else pd.DataFrame()

    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {required_columns}")
        return df
    
    def _load_csv(self, location: str, filename: str) -> pd.DataFrame:
        path = os.path.join(self.CENSUS_DATA, location.split(",")[0].strip().lower(), filename)
        return pd.read_csv(path)