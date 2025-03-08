from src.reporting.metadata import add_metadata_to_report
from src.reporting.profiler import generate_report
from src.population_generation.household_generator import generate_households
from src.llm_interface.ollama_model import OllamaModel
from src.file_loaders.prompt_loader import load_prompt
from src.file_loaders.schema_loader import load_schema
import time
import pandas as pd

model = OllamaModel("llama3.2:3b")
prompt = load_prompt("minimal_prompt.txt", {"city": "Newcastle, UK"})
schema = load_schema("household_schema.json")

n_households = 300

start_time = time.time()

try:
    households = generate_households(n_households, model, prompt, schema)

    execution_time = time.time() - start_time

    flat_data = [person for household in households for person in household]
    df = pd.DataFrame(flat_data)
    report_filename = generate_report(df)

    metadata = {
        "Execution Time": f"{execution_time:.2f} sec",
        "Number of Households": n_households,
        "Number of Individuals": len(flat_data),
        "Avg. Time per Household": f"{execution_time / n_households if n_households > 0 else 0:.4f} seconds",
        "Avg. Time per Person": f"{execution_time / len(flat_data) if len(flat_data) > 0 else 0:.4f} seconds",
        "Average Household Size": f"{len(flat_data) / n_households if n_households > 0 else 0:.2f}"
    }

    add_metadata_to_report(report_filename, metadata, prompt, model)

except Exception as e:
    print(f"An error occurred: {e}")