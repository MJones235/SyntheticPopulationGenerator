from datetime import datetime
import hashlib
import uuid
import pandas as pd
from tqdm import tqdm

from src.llm_interface.base_llm import BaseLLM
from src.repositories.estimation_metadata_repository import EstimationMetadataRepository
from src.repositories.estimation_repository import EstimationRepository
from src.services.file_service import FileService
from src.utils.number_validator import is_number


VARIABLE_CONFIG = {
    "population_size": {
        "prompt": "population_size.txt",
        "schema": "population_size_schema.json",
    },
    "age_distribution": {
        "prompt": "age_distribution.txt",
        "schema": "age_distribution_schema.json",
    },
}

class Estimator:
    def __init__(self, variable: str, model: BaseLLM, n_trials: int):
        self.variable = variable
        self.model = model
        self.n_trials = n_trials
        self.file_service = FileService()
        self.metadata_repo = EstimationMetadataRepository()
        self.estimation_repo = EstimationRepository()

        self.prompt_template = VARIABLE_CONFIG[variable]["prompt"]
        self.schema_name = VARIABLE_CONFIG[variable]["schema"]

        self.data_path = f"data/evaluation/{variable}/sampled_{variable}.csv"
        self.df = pd.read_csv(self.data_path)
        self.schema = self.file_service.load_schema(self.schema_name)

        self.run_id = str(uuid.uuid4())

    def run(self):
        input_hash = hashlib.sha256(self.df.to_csv(index=False).encode()).hexdigest()

        self.metadata_repo.insert_metadata({
            "run_id": self.run_id,
            "variable": self.variable,
            "model_name": self.model.model_name,
            "n_trials": self.n_trials,
            "prompt_template": self.prompt_template,
            "schema_name": self.schema_name,
            "input_hash": input_hash,
            "run_timestamp": datetime.now().isoformat()
        })

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc=f"Estimating {self.variable}"):
            location = row["BUA name"]
            category = row["BUA size classification"]
            gt = float(row["Counts"])
            prompt = self.file_service.load_prompt(self.prompt_template, {"input": location})

            for trial in range(self.n_trials):
                try:
                    response = self.model.generate_json(prompt, self.schema, n_attempts=1).get(self.variable, "N/A")
                    pred = float(response) if is_number(response) else None
                except Exception:
                    pred = None

                self.estimation_repo.insert_estimation({
                    "run_id": self.run_id,
                    "location": location,
                    "category": category,
                    "ground_truth": gt,
                    "trial_number": trial + 1,
                    "prediction": pred,
                    "timestamp": datetime.now().isoformat()
                })



