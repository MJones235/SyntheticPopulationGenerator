import concurrent
from src.llm_interface.model_factory import model_factory
from src.llm_interface.ollama_model import OllamaModel
from src.llm_interface.base_llm import BaseLLM

def generate_single_household(n: int, n_total: int, model_type:str, model_kwargs: dict, prompt: str, schema: str):
    print(f"Generating household {n + 1}/{n_total}")
    try:
        model = model_factory(model_type, **model_kwargs)
        result= model.generate_household(prompt, schema)
        return result["household"]
    except Exception as e:
        print("[ERROR] Error generating household. Skipping...")
        print(e)
        return None

def generate_households(n_households: int, model: BaseLLM, prompt: str, schema: str) -> list:
    model_type = "ollama" if isinstance(model, OllamaModel) else ""
    
    model_kwargs = {
        "model_name": model.model_name,
        "temperature": model.temperature,
        "top_p": model.top_p,
        "top_k": model.top_k
    }

    Executor = concurrent.futures.ProcessPoolExecutor if model.is_local else concurrent.futures.ThreadPoolExecutor

    with Executor() as executor:
        results = list(executor.map(generate_single_household, range(n_households), [n_households] * n_households,
            [model_type] * n_households, [model_kwargs] * n_households, [prompt] * n_households, [schema] * n_households))

    return [h for h in results if h is not None]