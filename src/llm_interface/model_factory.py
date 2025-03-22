from src.llm_interface.base_llm import BaseLLM
from src.llm_interface.ollama_model import OllamaModel
from src.llm_interface.openai_model import OpenAIModel
from src.llm_interface.gemini_model import GeminiModel

class LLMFactory:
    @staticmethod
    def get_provider(model_type: str, **kwargs) -> BaseLLM:
        if model_type == "ollama":
            return OllamaModel(**kwargs)
        elif model_type == "openai":
            return OpenAIModel(**kwargs)
        elif model_type == "gemini":
            return GeminiModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
