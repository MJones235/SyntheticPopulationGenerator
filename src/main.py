from src.repositories.metadata_repository import MetadataRepository
from src.repositories.population_repository import PopulationRepository
from src.reporting.profiler import generate_report
from src.population_generation.household_generator import generate_households
from src.llm_interface.ollama_model import OllamaModel
from src.file_loaders.prompt_loader import load_prompt
from src.file_loaders.schema_loader import load_schema
import time
import pandas as pd
import uuid

population_repo = PopulationRepository()
metadata_repo = MetadataRepository()

model = OllamaModel("llama3.2:3b", temperature=1, top_p=0.85, top_k=100)
location = "Newcastle, UK"
prompt = load_prompt("minimal_prompt.txt", {"city": location})
schema = load_schema("household_schema.json")

population_id = str(uuid.uuid4())

n_households = 10

start_time = time.time()

try:
    households = generate_households(n_households, model, prompt, schema)

    execution_time = time.time() - start_time

    flat_data = [person for household in households for person in household]
    df = pd.DataFrame(flat_data)
    report_filename = generate_report(population_id, df)

    metadata = {
        "population_id": population_id,
        "location": location,
        "model": model.model_name,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "top_k": model.top_k,
        "num_households": len(households),
        "execution_time": execution_time,
        "prompt": prompt,
    }

    metadata_repo.insert_metadata(metadata)
    population_repo.insert_population(population_id, households)


except Exception as e:
    print(f"An error occurred: {e}")