import subprocess
from src.llm_interface.base_llm import BaseLLM
from langchain_ollama import OllamaLLM
import multiprocessing

class OllamaModel(BaseLLM):
    is_local = True

    def __init__(self, model_name: str, temperature: float = 0.7, top_p: float = 0.95, top_k: int = 40, format: str = "json", **kwargs):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.format = format
        self.kwargs = kwargs
        self.llm = self._load_model()
    
    def get_model_metadata(self):
        return f'OllamaModel("{self.model_name}", temperature={self.temperature}, top_p={self.top_p}, top_k={self.top_k})'

    def generate_text(self, prompt: str | list[str], timeout=30) -> str | list[str]:
        def call_llm(queue):
            """Function to execute the LLM request inside a separate process."""
            try:
                if isinstance(prompt, str):
                    result = self.llm.invoke(prompt)
                else:
                    result = self.llm.batch(prompt)
                queue.put(result)  # Send result to main process
            except Exception as e:
                queue.put(e)  # Send error to main process

        queue = multiprocessing.Queue()
        process = multiprocessing.Process(target=call_llm, args=(queue,))
        process.start()
        process.join(timeout)

        if process.is_alive():
            print("[TIMEOUT] LLM call exceeded time limit. Terminating process...")
            process.terminate() 
            process.join()  
            raise TimeoutError(f"LLM batch call timed out after {timeout} seconds.")

        if not queue.empty():
            response = queue.get()
            if isinstance(response, Exception):
                raise response 
            return response

        raise TimeoutError("LLM call did not return a response.")
    
    def _load_model(self):
        available_models = self._get_available_models()
        if self.model_name not in available_models:
            print(f"[WARN] Model '{self.model_name}' is not installed. Attempting to pull...")
            if not self._pull_model():
                print(f"[ERROR] Could not pull model '{self.model_name}'.")
        return OllamaLLM(model=self.model_name, temperature=self.temperature, top_p=self.top_p, top_k=self.top_k, format=self.format, num_ctx=4096, num_predict=2048, num_thread=12, **self.kwargs)
            

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
        
            