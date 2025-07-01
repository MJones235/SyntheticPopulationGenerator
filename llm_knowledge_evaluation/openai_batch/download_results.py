import os
import json
import argparse
from datetime import datetime
import re
from dotenv import load_dotenv
import openai
from src.repositories.estimation_metadata_repository import EstimationMetadataRepository
from src.repositories.estimation_repository import EstimationRepository

load_dotenv("secrets.env")
openai.api_key = os.getenv("OPENAI_API_KEY")

def clean_json_block(raw: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\n?", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```$", "", cleaned.strip())
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", cleaned)
    return cleaned.strip()

def parse_and_insert(jsonl_path, metadata: dict):
    repo = EstimationRepository()
    lookup = metadata["metadata"]
    variable = metadata["variable"]

    with open(jsonl_path, "r") as f:
        for line in f:
            try:
                item = json.loads(line)
                custom_id = item["custom_id"]
                location = custom_id.split("_")[0]
                meta = lookup.get(custom_id, {})

                response = item.get("response", {})
                content = clean_json_block(response.get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", ""))

                try:
                    data = json.loads(content)
                    value = data.get("percentage")
                    pred = float(value) if isinstance(value, (int, float)) else None
                except json.JSONDecodeError:
                    print(f"[WARN] Could not parse JSON for {custom_id}: {content}")
                    pred = None

                repo.insert_estimation({
                    "run_id": metadata["run_id"],
                    "variable": variable,
                    "location": location,
                    "category": meta.get("category"),
                    "subcategory": meta.get("subcategory"),
                    "ground_truth": meta.get("ground_truth"),
                    "trial_number": 1,
                    "prediction": pred,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                print(f"[WARN] Skipping record due to error: {e}")

def insert_metadata(metadata: dict):
    meta_repo = EstimationMetadataRepository()

    input_hash = metadata.get("input_hash", None)

    meta_repo.insert_metadata({
        "run_id": metadata["run_id"],
        "variable": metadata["variable"],
        "model_name": metadata["model"],
        "n_trials": metadata.get("n_trials", 1),
        "prompt_template": metadata.get("prompt_template", "unknown"),
        "schema_name": metadata.get("schema_name", "unknown"),
        "input_hash": input_hash,
        "run_timestamp": datetime.now().isoformat()
    })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, help="Path to batch metadata JSON file")
    args = parser.parse_args()

    with open(args.metadata, "r") as f:
        metadata = json.load(f)

    batch_id = metadata["batch_id"]
    batch = openai.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"⏳ Batch {batch_id} is not yet complete. Current status: {batch.status}")
        return

    print(f"✅ Batch complete. Output file ID: {batch.output_file_id}")

    # Download and save the result file
    result_stream = openai.files.content(batch.output_file_id)
    result_path = args.metadata.replace("_metadata.json", "_results.jsonl")

    with open(result_path, "wb") as f:
        f.write(result_stream.read())

    print(f"📥 Results downloaded to {result_path}")
    parse_and_insert(result_path, metadata)
    insert_metadata(metadata)
    print(f"✅ All predictions inserted into the database.")

if __name__ == "__main__":
    main()