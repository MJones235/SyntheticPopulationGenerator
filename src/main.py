from src.services.analysis_service import AnalysisService
from src.services.metadata_service import MetadataService
from src.services.report_service import ReportService
from src.services.population_service import PopulationService
from src.services.file_service import FileService
from src.repositories.metadata_repository import MetadataRepository
from src.repositories.population_repository import PopulationRepository
from src.llm_interface.ollama_model import OllamaModel
import time
import pandas as pd
import uuid

file_service = FileService()
population_service = PopulationService()
report_service = ReportService()
metadata_service = MetadataService()
analysis_service = AnalysisService()

model = OllamaModel("phi3:14b", temperature=1, top_p=0.85, top_k=100)
location = "Newcastle, UK"
prompt = file_service.load_prompt("minimal_prompt.txt", {"LOCATION": location})
schema = file_service.load_schema("household_schema.json")

population_id = str(uuid.uuid4())

n_households = 300
batch_size = 10

start_time = time.time()

try:
    households = population_service.generate_households(n_households, model, prompt, schema, batch_size, location)

    execution_time = time.time() - start_time

    flat_data = [person for household in households for person in household]
    df = pd.DataFrame(flat_data)
    report_filename = report_service.generate_report(population_id, df)

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

    metadata_service.save_metadata(metadata)
    population_service.save_population(population_id, households)
    analysis_service.save_analysis(population_id)

except Exception as e:
    print(f"An error occurred: {e}")