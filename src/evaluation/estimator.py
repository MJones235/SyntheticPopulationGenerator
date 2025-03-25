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
        self.input_hash = hashlib.sha256(self.df.to_csv(index=False).encode()).hexdigest()
        self.schema = self.file_service.load_schema(self.schema_name)

        self.run_id = str(uuid.uuid4())

    def run(self):
        self.metadata_repo.insert_metadata({
            "run_id": self.run_id,
            "variable": self.variable,
            "model_name": self.model.model_name,
            "n_trials": self.n_trials,
            "prompt_template": self.prompt_template,
            "schema_name": self.schema_name,
            "input_hash": self.input_hash,
            "run_timestamp": datetime.now().isoformat()
        })

        prompts, metadata = self.get_batch_prompts_and_metadata()

        for custom_id, prompt in tqdm(prompts, desc=f"Estimating {self.variable}"):
            location = custom_id.split("_")[0]
            meta = metadata.get(custom_id, {})
            for trial in range(self.n_trials):
                try:
                    response = self.model.generate_json(prompt, self.schema, n_attempts=1)
                    value = response.get(self.variable) if self.variable == "population_size" else response.get("percentage")
                    pred = float(value) if is_number(value) else None
                except Exception:
                    pred = None

                self.estimation_repo.insert_estimation({
                    "run_id": self.run_id,
                    "variable": self.variable,
                    "location": location,
                    "category": meta.get("category"),
                    "subcategory": meta.get("subcategory"),
                    "ground_truth": meta.get("ground_truth"),
                    "trial_number": trial + 1,
                    "prediction": pred,
                    "timestamp": datetime.now().isoformat()
                })


    def get_batch_prompts_and_metadata(self) -> tuple[list[tuple[str, str]], dict[str, dict]]:
        prompts = []
        metadata = {}

        if self.variable == "population_size":
            for i, row in self.df.iterrows():
                location = row["BUA name"]
                category = row["BUA size classification"]
                gt = float(row["Counts"]) if is_number(row["Counts"]) else None
                custom_id = f"{location}_{i}"

                prompt = self.file_service.load_prompt(self.prompt_template, {"input": location})
                prompts.append((custom_id, prompt))
                metadata[custom_id] = {
                    "category": category,
                    "subcategory": None,
                    "ground_truth": gt
                }

        elif self.variable == "age_distribution":
            metadata_cols = ["BUA name", "Region", "Country", "BUA size classification"]
            value_cols = [col for col in self.df.columns if col not in metadata_cols]

            for i, row in self.df.iterrows():
                location = row["BUA name"]
                category = row["BUA size classification"]

                for col in value_cols:
                    try:
                        age_band, sex = col.rsplit(" ", 1)
                    except ValueError:
                        continue

                    subcategory = f"{age_band} {sex}"
                    gt = float(row[col]) if is_number(row[col]) else None
                    custom_id = f"{location}_{subcategory}_{i}"

                    prompt_input = {
                        "LOCATION": location,
                        "AGE": age_band,
                        "SEX": sex
                    }
                    prompt = self.file_service.load_prompt(self.prompt_template, prompt_input)

                    prompts.append((custom_id, prompt))
                    metadata[custom_id] = {
                        "category": category,
                        "subcategory": subcategory,
                        "ground_truth": gt
                    }

        return prompts, metadata
