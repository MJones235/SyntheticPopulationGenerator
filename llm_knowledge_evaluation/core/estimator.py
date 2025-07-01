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
    "age_distribution": {
        "prompt": "age_distribution.txt",
        "schema": "evaluation_schema.json",
        "subcategory_col": "Age Bin",
        "prompt_inputs": lambda row: {
            "LOCATION": row["Upper tier local authorities"],
            "AGE_BAND": row["Age Bin"]
        },
    },
    "household_size": {
        "prompt": "household_size.txt",
        "schema": "evaluation_schema.json",
        "subcategory_col": "Household size (9 categories)",
        "prompt_inputs": lambda row: {
            "LOCATION": row["Upper tier local authorities"],
            "HOUSEHOLD_SIZE_DESCRIPTION": row["Household size (9 categories)"]
        },
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

        self.data_path = f"data/evaluation/{variable}/sampled_data.csv"
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

        config = VARIABLE_CONFIG.get(self.variable)
        if not config:
            return prompts, metadata
        
        subcategory_col = config["subcategory_col"]
        make_prompt_input = config["prompt_inputs"]

        for i, row in self.df.iterrows():
            location = row["Upper tier local authorities"]
            subcategory = row[subcategory_col]
            gt = float(row["Percentage"]) if is_number(row["Percentage"]) else None
            custom_id = f"{location}_{subcategory}_{i}"

            prompt_input = make_prompt_input(row)
            prompt = self.file_service.load_prompt(self.prompt_template, prompt_input)
            prompts.append((custom_id, prompt))
            metadata[custom_id] = {
                "category": None,
                "subcategory": subcategory,
                "ground_truth": gt
            }

        return prompts, metadata