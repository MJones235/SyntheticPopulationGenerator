import os
import json
from typing import Dict
import pandas as pd
import re

class FileService:
    PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/schemas/")
    CENSUS_DATA = os.path.join(os.path.dirname(__file__), "../../data/locations/")

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
