import os
from dotenv import load_dotenv
from src.llm_interface.base_llm import BaseLLM
from src.llm_interface.ollama_model import OllamaModel
from src.llm_interface.openai_model import OpenAIModel
from src.llm_interface.gemini_model import GeminiModel

class LLMFactory:
    @staticmethod
    def get_provider(model_type: str, **kwargs) -> BaseLLM:
        load_dotenv("secrets.env")

        if model_type == "ollama":
            return OllamaModel(**kwargs)
        elif model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing OPENAI_API_KEY in secrets.env")
            return OpenAIModel(api_key=api_key, **kwargs)
        elif model_type == "gemini":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise EnvironmentError("Missing GEMINI_API_KEY in secrets.env")
            return GeminiModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
