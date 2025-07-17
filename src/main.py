import os
from dotenv import load_dotenv
from llm_interface.azure_model import AzureModel
from llm_interface.openai_model import OpenAIModel
from src.services.experiment_run_service import ExperimentRunService
from src.services.experiments_service import ExperimentService
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
experiments_service = ExperimentService()
experiment_run_service = ExperimentRunService()

# model = OllamaModel("llama3.1:8b", temperature=0.7, top_p=0.85, top_k=100)
load_dotenv("secrets.env")

model = AzureModel(
    model_name="Phi-4-reasoning",
    api_key=os.getenv("AZURE_API_KEY"),
    temperature=0.7,
    top_p=0.85,
    top_k=100,
)
#model = OpenAIModel(
#    model_name="gpt-4o",
#    api_key=os.getenv("OPENAI_API_KEY"),
#    temperature=0.7,
#    top_p=0.85,
#    top_k=100
#)

location = "Newcastle, UK"
region = "E12000001"
n_households = 100
batch_size = 10
include_stats = True
include_target = False
include_guidance = False
compute_household_size = False
use_microdata = False
no_occupation = True
no_household_composition = True

if use_microdata:
    prompt_file = "microdata.txt"
elif compute_household_size:
    prompt_file = "fixed_household_size.txt"
elif no_occupation:
    prompt_file = "no_occupation.txt"
else:
    prompt_file = "minimal_prompt.txt"

prompt = file_service.load_prompt(
    prompt_file, {"LOCATION": location, "TOTAL_HOUSEHOLDS": str(n_households)}
)

if no_occupation:
    schema = file_service.load_schema("household_schema_no_occupation.json")
else:
    schema = file_service.load_schema("household_schema.json")


n_runs = 1
experiment_id = str(uuid.uuid4())
experiment_start_time = time.time()

for run in range(n_runs):
    population_id = str(uuid.uuid4())

    start_time = time.time()

    try:
        households = population_service.generate_households(
            n_households,
            model,
            prompt,
            schema,
            location,
            region,
            batch_size,
            include_stats,
            include_guidance,
            use_microdata,
            compute_household_size,
            include_target,
            no_occupation,
            run+1,
            no_household_composition
        )
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
            "include_stats": include_stats,
            "include_guidance": include_guidance,
            "include_target": include_target,
            "use_microdata": use_microdata,
            "compute_household_size": compute_household_size,
            "no_occupation": no_occupation,
            "no_household_composition": no_household_composition,
        }

        run = {
            "experiment_id": experiment_id,
            "run_number": run,
            "population_id": population_id,
            "execution_time": execution_time,
        }

        metadata_service.save_metadata(metadata)
        population_service.save_population(population_id, households)
        analysis_service.save_analysis(population_id)
        experiment_run_service.save_run(run)

    except Exception as e:
        print(f"An error occurred: {e}")

experiment_execution_time = experiment_start_time - time.time()
experiment = {
    "experiment_id": experiment_id,
    "location": location,
    "model": model.model_name,
    "temperature": model.temperature,
    "top_p": model.top_p,
    "top_k": model.top_k,
    "execution_time": experiment_execution_time,
    "prompt": prompt,
    "include_stats": include_stats,
    "include_guidance": include_guidance,
    "include_target": include_target,
    "use_microdata": use_microdata,
    "compute_household_size": compute_household_size,
    "no_occupation": no_occupation,
    "no_household_composition": no_household_composition,
}

experiments_service.save_experiment(experiment)