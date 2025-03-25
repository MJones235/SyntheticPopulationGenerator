import traceback
from llm_interface.model_factory import LLMFactory
from src.evaluation.estimator import Estimator
from src.llm_interface.ollama_model import OllamaModel

VARIABLES = [
    # "population_size"
    "age_distribution"
]

MODELS = [
    {"name": "llama3.1:8b", "type": "ollama"},
    {"name": "llama3.2:3b", "type": "ollama"},
    {"name": "deepseek-r1:7b", "type": "ollama"},
    {"name": "gemma2:9b", "type": "ollama"},
    {"name": "mistral:latest", "type": "ollama"},
    {"name": "phi3:14b", "type": "ollama"},
    {"name": "qwen2.5:14b", "type": "ollama"},
    {"name": "qwen2.5:7b", "type": "ollama"},
]

def run_batch():
    for variable in VARIABLES:
        for model_cfg in MODELS:
            try:
                print(f"\n🧪 Running estimation: {model_cfg['name']} on {variable}")
                model = LLMFactory.get_provider(model_cfg["type"], model_name=model_cfg["name"], temperature=0)
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