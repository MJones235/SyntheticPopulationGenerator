import pandas as pd
import os
import datetime
import uuid
import json

from tqdm import tqdm

from src.services.file_service import FileService
from src.llm_interface.ollama_model import OllamaModel

###############
# SCRIPT CONFIG
###############

var_name = "population_size"
model = OllamaModel("deepseek-r1:7b", temperature=0)
n_trials = 5

###############
# LOAD SERVICES
###############

file_service = FileService()

###############
# LOAD INPUTS
###############

DIR = os.path.join(os.path.dirname(__file__), f"../../data/evaluation/{var_name}")
input_filepath = os.path.join(DIR, f"sampled_{var_name}.csv")

if not os.path.exists(input_filepath):
    raise FileNotFoundError(f"File not found: {input_filepath}")

df = pd.read_csv(input_filepath)

schema = file_service.load_schema(f"{var_name}_schema.json")

###############
# GENERATE DATA
###############

results = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Querying LLM"):
    location = row["BUA name"]
    prompt = file_service.load_prompt(f"{var_name}.txt", {"input": location})
    print(location)
    responses = []
    for _ in range(n_trials):
        try:
            response = model.generate_json(prompt, schema, n_attempts=1, timeout=10).get(var_name, "N/A")
        except Exception:
            response = "N/A"
        responses.append(response)
    
    result = {
        "Country": row["Country"],
        "Region": row["Region"],
        "BUA name": location,
        "BUA size classification": row["BUA size classification"],
        "Counts": row["Counts"]
    }

    for i in range(n_trials):
        result[f"Counts_{i + 1}"] = responses[i] if i < len(responses) else "N/A"

    results.append(result)

df_results = pd.DataFrame(results)

###############
# SAVE OUTPUTS
###############

output_filepath = file_service.generate_unique_filename(DIR, f"estimated_{var_name}.csv")
metadata_filepath = output_filepath.replace(".csv", "_metadata.json")

metadata = {
    "unique_id": str(uuid.uuid4()),
    "model_name": model.model_name,
    "model_metadata": model.get_model_metadata(),
    "n_trials": n_trials,
    "input_file": os.path.basename(input_filepath),
    "ouput_file": os.path.basename(output_filepath),
    "timestamp": datetime.datetime.now().isoformat(),
    "prompt_template": f"{var_name}.txt"
}

df_results.to_csv(output_filepath, index=False)

with open(metadata_filepath, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"LLM queries completed with {n_trials} trials per location. Results saved to {output_filepath}.")