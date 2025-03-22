from src.llm_interface.base_llm import BaseLLM
import google.generativeai as genai

class GeminiModel(BaseLLM):
    is_local = False

    def __init__(self, model_name: str = "gemini-pro", api_key: str = None, temperature: float = 0.7, **kwargs):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self.temperature = temperature

    def get_model_metadata(self) -> str:
        return f'GeminiModel("{self.model_name}")'

    def generate_text(self, prompt: str | list[str], timeout=30) -> str | list[str]:
        if isinstance(prompt, list):
            return [self._call_gemini(p, timeout) for p in prompt]
        return self._call_gemini(prompt, timeout)

    def _call_gemini(self, prompt: str, timeout=30) -> str:
        response = self.model.generate_content(prompt, generation_config={"temperature": self.temperature})
        return response.text.strip()
