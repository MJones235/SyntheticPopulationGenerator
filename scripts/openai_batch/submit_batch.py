import os
import json
from datetime import datetime
import openai
from dotenv import load_dotenv
from src.evaluation.estimator import Estimator

# --- Config
VARIABLE = "household_size"
MODEL = "gpt-4o"
TEMPERATURE = 0.0

BATCH_PART = 1
MAX_PROMPTS = 450

# --- Setup
load_dotenv("secrets.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Initialize Estimator (only used for prompt/metadata generation)
estimator = Estimator(variable=VARIABLE, model=None, n_trials=1)
prompts, metadata_dict = estimator.get_batch_prompts_and_metadata()


start_idx = (BATCH_PART - 1) * MAX_PROMPTS
end_idx = start_idx + MAX_PROMPTS
prompts = prompts[start_idx:end_idx]

# --- Prepare batch output paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
basename = f"{VARIABLE}_{MODEL}_part{BATCH_PART}_{timestamp}"
jsonl_path = f"outputs/batch_jobs/{basename}.jsonl"
metadata_path = f"outputs/batch_jobs/{basename}_metadata.json"

# --- Write JSONL
with open(jsonl_path, "w") as f:
    for custom_id, prompt in prompts:
        f.write(json.dumps({
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": TEMPERATURE
            }
        }) + "\n")

print(f"✅ JSONL file ready: {jsonl_path}")

# --- Submit to OpenAI
upload = openai.files.create(file=open(jsonl_path, "rb"), purpose="batch")
batch = openai.batches.create(
    input_file_id=upload.id,
    endpoint="/v1/chat/completions",
    completion_window="24h"
)

print(f"🚀 Batch submitted: {batch.id} | Status: {batch.status}")

# --- Save metadata
metadata = {
    "run_id": estimator.run_id,
    "variable": VARIABLE,
    "n_trials": estimator.n_trials,
    "prompt_template": estimator.prompt_template,
    "schema_name": estimator.schema_name,
    "input_hash": estimator.input_hash,
    "model": MODEL,
    "batch_id": batch.id,
    "metadata": metadata_dict
}

os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"📦 Metadata saved to: {metadata_path}")
