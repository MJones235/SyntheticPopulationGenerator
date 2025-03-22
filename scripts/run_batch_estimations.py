import traceback
from src.evaluation.estimator import Estimator
from src.llm_interface.ollama_model import OllamaModel

VARIABLES = ["population_size"]

MODELS = [
    {"name": "deepseek-r1:7b"},
    {"name": "gemma2:9b"},
    {"name": "llama3.1:8b"},
    {"name": "llama3.2:3b"},
    {"name": "mistral:latest"},
    {"name": "phi3:14b"},
    {"name": "qwen2.5:14b"},
    {"name": "qwen2.5:7b"},
]

def run_batch():
    for variable in VARIABLES:
        for model_cfg in MODELS:
            try:
                print(f"\n🧪 Running estimation: {model_cfg['name']} on {variable}")
                model = OllamaModel(model_cfg["name"], temperature=0)
                n_trials = 1

                estimator = Estimator(
                    variable=variable,
                    model=model,
                    n_trials=n_trials
                )
                estimator.run()
                print(f"✅ Completed: {model_cfg['name']} on {variable}")
            except Exception as e:
                print(f"❌ Failed: {model_cfg['name']} on {variable}")
                traceback.print_exc()

if __name__ == "__main__":
    run_batch()