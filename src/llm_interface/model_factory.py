from src.llm_interface.base_llm import BaseLLM
from src.llm_interface.ollama_model import OllamaModel

class LLMFactory:
    @staticmethod
    def get_provider(model_type: str, **kwargs) -> BaseLLM:
        if model_type == "ollama":
            return OllamaModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
