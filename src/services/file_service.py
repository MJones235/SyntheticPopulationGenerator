import os
import json
from typing import Optional, Union
import pandas as pd
import re

class FileService:
    PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/schemas/")
    CENSUS_DATA = os.path.join(os.path.dirname(__file__), "../../data/aggregate/processed/")
    MICRODATA = os.path.join(os.path.dirname(__file__), "../../data/microdata/household_uk.tab")

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

    def load_household_size(self, location: str) -> dict:
        return self._load_csv(location, "household_size.csv", 0)
        
    def load_age_distribution(self, location: str) -> dict:
        """Loads age distribution data aggregated across genders."""
        try:
            path = os.path.join(self.CENSUS_DATA, location.split(",")[0].replace(" ", "_").strip().lower(), "age_group.csv")
            df = pd.read_csv(path)
            
            # Aggregate percentages by age group (Category_1), summing across genders
            age_dist = df.groupby("Category_1")["Percentage"].sum().to_dict()
            
            return age_dist
        except Exception as e:
            print(f"Error loading age distribution for {location}: {e}")
            return {}
        
    def load_household_composition(self, location: str) -> dict:
        return self._load_csv(location, "household_composition.csv")

    def load_age_pyramid(self, location: str) -> pd.DataFrame:
        try:
            path = os.path.join(self.CENSUS_DATA, location.split(",")[0].replace(" ", "_").strip().lower(), "age_group.csv")
            df = pd.read_csv(path)

            pyramid_df = df.pivot_table(
                index="Category_1",
                columns="Category_2",
                values="Percentage",
                aggfunc="sum"
            ).fillna(0)

            pyramid_df.columns = [c.strip().capitalize() for c in pyramid_df.columns]
            pyramid_df = pyramid_df.reindex(columns=["Male", "Female"], fill_value=0)

            def sort_key(label):
                match = re.match(r"(\d+)", str(label))
                return int(match.group(1)) if match else float("inf")

            pyramid_df = pyramid_df.sort_index(key=lambda x: x.map(sort_key))
            pyramid_df = pyramid_df.reset_index().rename(columns={"Category_1": "age_group"})

            return pyramid_df

        except Exception as e:
            print(f"Failed to load age pyramid for {location}: {e}")
            return pd.DataFrame(columns=["Male", "Female"])

    def load_occupation_distribution(self, location: str) -> dict:
        return self._load_csv(location, "occupation.csv")
    
    def load_sex_distribution(self, location: str) -> dict:
        return self._load_csv(location, "sex.csv")

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
            df = pd.read_csv(self.MICRODATA, sep="\t", dtype={"iol22cd": str})
            df = df[df["region"] == region]
            return df
        except Exception as e:
            print(f"Error loading microdata for {region}: {e}")
            return pd.DataFrame()
        
    def load_avg_household_size(self, location: str) -> float:
        try:
            df = pd.read_csv(os.path.join(self.CENSUS_DATA, location.split(",")[0].replace(" ", "_").strip().lower(), "avg_household_size.csv"))
            if "Value" in df.columns:
                return df["Value"].iloc[0]
            else:
                raise KeyError("Expected column 'Value' not found in avg_household_size.csv")
        except Exception as e:
            print(f"Error loading average household size for {location}: {e}")
            return 0.0
        
    def load_partner_age_diff(self, location: str) -> dict:
        return self._load_csv(location, "partner_age_diff.csv")
    
    def _load_csv(self, location: str, filename: str, exclude_category_1: Optional[Union[str, int, float]] = None) -> pd.DataFrame:
        path = os.path.join(
            self.CENSUS_DATA,
            location.split(",")[0].replace(" ", "_").strip().lower(),
            filename
        )
        df = pd.read_csv(path)

        if exclude_category_1 is not None:
            df = df[df["Category_1"] != exclude_category_1]

        return dict(zip(df["Category_1"], df["Percentage"]))
