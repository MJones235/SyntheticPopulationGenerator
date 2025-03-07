import os

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "../prompts")

def load_prompt(filename: str, replacements: dict = None) -> str:
    """
    Load a prompt file from the `prompts/` directory and optionally replace placeholders.

    :param filename: Name of the prompt file (e.g., "minimal_prompt.txt")
    :param replacements: Dictionary of placeholders to replace (e.g., {"city": "Newcastle"})
    :return: The formatted prompt string
    """
    filepath = os.path.join(PROMPT_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Prompt file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as file:
        prompt = file.read()

    if replacements:
        for key, value in replacements.items():
            prompt = prompt.replace(f"{{{key}}}", str(value))

    return prompt
