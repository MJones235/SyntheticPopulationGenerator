import os
import json

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../../data/schemas/")

def load_schema(filename: str):
    """Loads the household validation schema from file."""
    filepath = os.path.join(SCHEMA_PATH, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)
