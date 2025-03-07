import json
import subprocess
from src.llm_interface.base_llm import BaseLLM
from langchain_ollama import OllamaLLM

class OllamaModel(BaseLLM):
    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, max_retries: int = 3, format: str = "json", **kwargs):
        super().__init__(max_retries=max_retries)
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.format = format
        self.kwargs = kwargs
        self.llm = self._load_model()

    def generate_text(self, prompt):
        response = self.llm.invoke(prompt)
        return json.loads(response) if format == "json" else response
    
    def _load_model(self):
        available_models = self._get_available_models()
        if self.model_name not in available_models:
            print(f"[WARN] Model '{self.model_name}' is not installed. Attempting to pull...")
            if not self._pull_model():
                print(f"[ERROR] Could not pull model '{self.model_name}'.")
        return OllamaLLM(model=self.model_name, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, format=self.format, **self.kwargs)
            

    def _pull_model(self) -> bool:
        cmd_list = ["ollama", "pull", self.model_name]
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[WARN] Could not pull model: {result.stderr}")
                return False
            return True
        except Exception as e:
            print(f"[WARN] Could not pull model: {e}")
            return False

    def _get_available_models(self):
        cmd_list = ["ollama", "list"]
        try:
            result = subprocess.run(cmd_list, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"[WARN] Could not list models: {result.stderr}")
                return []
            
            return [line.split()[0] for line in result.stdout.splitlines()[1:] if line.strip()]

        except Exception as e:
            print(f"[WARN] Could not list models: {e}")
            return []
        
            