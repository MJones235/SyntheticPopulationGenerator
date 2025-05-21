import os
import json
from typing import Dict
import pandas as pd
import re

class FileService:
    PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/schemas/")
    CENSUS_DATA = os.path.join(os.path.dirname(__file__), "../../data/locations/")
    MICRODATA = os.path.join(os.path.dirname(__file__), "../../data/microdata/publicmicrodatateachingsample.csv")

    def load_prompt(self, filename: str, replacements: dict = None) -> str:
        """
        Load a prompt file from the `prompts/` directory and optionally replace placeholders.

        :param filename: Name of the prompt file (e.g., "minimal_prompt.txt")
        :param replacements: Dictionary of placeholders to replace (e.g., {"city": "Newcastle"})
        :return: The formatted prompt string
        """
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
        try:
            
            filepath = os.path.join(self.CENSUS_DATA, location.split(",")[0].strip().lower(), "household_size.csv")
            census_df = pd.read_csv(filepath)
            required_columns = ["C2021_HHSIZE_10_NAME", "OBS_VALUE"]
            if not all(col in census_df.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")

            census_df["Household Size"] = census_df["C2021_HHSIZE_10_NAME"].apply(lambda x: int(re.search(r"\d+", x).group()))
            household_size_distribution = dict(zip(census_df["Household Size"], census_df["OBS_VALUE"]))
            return household_size_distribution

        except Exception as e:
            print(f"Error processing census data: {e}")
            return {}
        
    def load_household_composition(self, location: str) -> Dict:
        try:
            
            filepath = os.path.join(self.CENSUS_DATA, location.split(",")[0].strip().lower(), "household_composition.csv")
            census_df = pd.read_csv(filepath)
            required_columns = ["Household composition (8 categories)", "Observation"]
            if not all(col in census_df.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")

            census_df["Household Composition"] = census_df["Household composition (8 categories)"]
            total = census_df["Observation"].sum()
            census_df["Value"] = census_df["Observation"].apply(lambda x: x / total * 100)
            return census_df[["Household Composition", "Value"]]

        except Exception as e:
            print(f"Error processing census data: {e}")
            return {}


    def load_age_pyramid(self, location: str) -> pd.DataFrame:
        try:
            csv_path = os.path.join(self.CENSUS_DATA, location.split(",")[0].strip().lower(), "age_group.csv")
            df = pd.read_csv(csv_path)

            # Clean column names
            df.columns = df.columns.str.strip()
            df["Percentage per BUA"] = df["Percentage per BUA"].astype(float)

            # Clean up age group labels
            df["Age"] = (
                df["Age"]
                .str.replace(r"Aged (\d+) years and under", r"0–\1", regex=True)
                .str.replace(r"Aged (\d+) to (\d+) years", r"\1–\2", regex=True)
                .str.replace(r"Aged (\d+) years and over", r"\1+", regex=True)
                .str.strip()
            )

            # Pivot into age_group × sex → percentage
            pyramid_df = df.pivot_table(index="Age", columns="Sex", values="Percentage per BUA", aggfunc="sum").fillna(0)

            # Standardize sex column names
            pyramid_df.columns = [c.strip().capitalize() for c in pyramid_df.columns]
            pyramid_df = pyramid_df.reindex(columns=["Male", "Female"], fill_value=0)

            # Sort by age group
            def sort_key(label):
                import re
                match = re.match(r"(\d+)", label)
                return int(match.group(1)) if match else float("inf")

            pyramid_df = pyramid_df.sort_index(key=lambda x: x.map(sort_key))
            pyramid_df = pyramid_df.reset_index().rename(columns={"Age": "age_group"})

            return pyramid_df

        except Exception as e:
            print(f"Failed to load age pyramid for {location}: {e}")
            return pd.DataFrame(columns=["Male", "Female"])

    def load_occupation_distribution(self, location: str) -> dict:
        try:
            csv_path = os.path.join(self.CENSUS_DATA, location.split(",")[0].strip().lower(), "occupation.csv")
            df = pd.read_csv(csv_path)

            # Rename for consistency
            df = df.rename(columns={
                "Occupation (current) (10 categories) Code": "occupation_category_code",
                "Observation": "value"
            })

            df["percentage"] = df["value"] / df["value"].sum() * 100
            df["percentage"] = df["percentage"].round(1)

            occupation_distribution = dict(zip(df["occupation_category_code"], df["percentage"]))
            return occupation_distribution

        except Exception as e:
            print(f"Failed to load occupation distribution: {e}")
            return {}



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

