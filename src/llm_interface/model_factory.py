from src.llm_interface.ollama_model import OllamaModel

def model_factory(model_type: str, **kwargs):
    """Factory function to create the appropriate LLM model."""
    if model_type == "ollama":
        return OllamaModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
