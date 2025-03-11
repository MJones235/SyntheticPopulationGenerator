import os
import json

class FileService:
    PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")
    SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/schemas/")

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
