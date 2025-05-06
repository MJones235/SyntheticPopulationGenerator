from openai import OpenAI
from src.llm_interface.base_llm import BaseLLM

class OpenAIModel(BaseLLM):
    is_local = False

    def __init__(self, model_name: str = "gpt-4", api_key: str = None, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, **kwargs):
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.kwargs = kwargs

    def get_model_metadata(self) -> str:
        return f'OpenAIModel("{self.model_name}")'

    def generate_text(self, prompt: str | list[str], timeout=30) -> str | list[str]:
        if isinstance(prompt, list):
            return [self._call_openai(p, timeout) for p in prompt]
        return self._call_openai(prompt, timeout)

    def _call_openai(self, prompt: str, timeout=30) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            timeout=timeout
        )
        return response.choices[0].message.content.strip()
