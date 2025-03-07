from src.llm_interface.ollama_model import OllamaModel
from src.utils.prompt_loader import load_prompt

model = OllamaModel("llama3.2:3b")
prompt = load_prompt("minimal_prompt.txt", {"city": "Newcastle, UK"})
result= model.generate_text(prompt)
print(result)