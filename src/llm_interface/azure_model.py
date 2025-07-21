from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from src.llm_interface.base_llm import BaseLLM
from azure.ai.inference.models import SystemMessage, UserMessage
import re

class AzureModel(BaseLLM):
    is_local = False

    def __init__(self, model_name: str = "DeepSeek-R1-0528", api_key: str = None, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, **kwargs):
        self.model_name = model_name
        self.client = ChatCompletionsClient(
            endpoint="https://population-generator-resource.services.ai.azure.com/models",
            credential=AzureKeyCredential(api_key),
            api_version="2024-05-01-preview"
        )
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.kwargs = kwargs

    def get_model_metadata(self) -> str:
        return f'AzureModel("{self.model_name}")'

    def generate_text(self, prompt: str | list[str], timeout=30) -> str | list[str]:
        if isinstance(prompt, list):
            return [self._call_azure(p) for p in prompt]
        return self._call_azure(prompt)

    def _call_azure(self, prompt: str) -> str:
        response = self.client.complete(
            model=self.model_name,
            messages=[
                SystemMessage("You are an expert demographic modeller generating realistic synthetic households for population simulation. Your goal is to produce one new household at a time, ensuring that the characteristics of each household and its members are plausible and reflect the statistical context provided."),
                UserMessage(prompt)
            ],
            temperature=self.temperature,
            top_p=self.top_p
        )

        print(response.usage)
        
        raw_output = response.choices[0].message.content.strip()
        return re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()

