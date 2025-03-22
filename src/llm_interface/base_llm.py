from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import jsonschema
import time

class BaseLLM(ABC):
    """
    An abstract base class defining the standard interface
    for local or API-based LLMs.
    It includes logic for generating JSON-based outputs,
    validating them, and retrying if necessary.
    """

    is_local = False
    model_name: str
    temperature: float

    @abstractmethod
    def generate_text(self, prompt: str | list[str], timeout: int) -> str | list[str]:
        """
        Subclasses must implement how to call the LLM and return a raw text response.
        """
        raise NotImplementedError
        
    @abstractmethod
    def get_model_metadata(self) -> str:
        raise NotImplementedError

    def generate_json(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        n_attempts: int = 3,
        timeout: int = 30
    ) -> Dict[str, Any]:

        attempts = 0
        current_prompt = prompt

        while attempts < n_attempts:
            try:
                raw_response = self.generate_text(current_prompt, timeout).strip()
            except Exception as e:
                attempts += 1
                print(f"Failed to generate response: {str(e)}.  Retrying...")
                continue

            try:
                data = json.loads(raw_response)
                jsonschema.validate(instance=data, schema=json_schema)
                return data

            except Exception as e:
                attempts += 1
                error_msg = f"JSON parse error: {str(e)}"
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    previous_output=raw_response,
                    error=error_msg,
                    schema=json_schema,
                )
                print("Unable to parse response.  Retrying...")
                continue 

        return []

    def generate_batch_json(self, prompts: List[str], json_schema: Dict[str, Any], max_parallel=4, n_attempts: int = 3, timeout=30) -> List[Dict[str, Any]]:

        failed_prompts = list(prompts)
        valid_responses = []

        for attempt in range(n_attempts):
            if not failed_prompts:
                break

            new_failed_prompts = []
            batch_start = time.time()

            for i in range(0, len(failed_prompts), max_parallel):
                print(f"[INFO] {"Generating" if attempt == 0 else "Regenerating"} households {i + 1} to {i + max_parallel}")
                batch_prompts = failed_prompts[i : i + max_parallel] 
                try:
                    batch_responses = self.generate_text(batch_prompts, timeout)
                except Exception as e:
                    new_failed_prompts.extend(batch_prompts) 
                    continue

                for prompt, response in zip(batch_prompts, batch_responses):
                    if response is None:
                        print(f"[WARNING] Missing response. Retrying...")
                        new_failed_prompts.append(prompt)
                        continue

                    try:
                        data = json.loads(response)
                        jsonschema.validate(instance=data, schema=json_schema)
                        valid_responses.append(data["household"])  # Extract and extend
                    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                        print(f"[ERROR] Response validation failed. Retrying...")
                        new_failed_prompts.append(self._build_correction_prompt(prompt, response, f"Validation error: {e}", json_schema))

            failed_prompts = new_failed_prompts
            batch_end = time.time()
            print(f"[INFO] Batch completed in {batch_end - batch_start:.2f} seconds.\n\n")

        return valid_responses


    def _build_correction_prompt(
        self,
        original_prompt: str,
        previous_output: str,
        error: str,
        schema: Dict[str, Any]
    ) -> str:
        """
        Builds a follow-up prompt to correct invalid JSON or schema violations.
        """
        corrected_prompt = (
            f"{original_prompt}\n\n"
            "The previous response was invalid JSON or didn't match the schema.\n"
            f"Original error: {error}\n"
            f"Please correct your response so it is valid JSON and matches this schema:\n{json.dumps(schema, indent=2)}"
            f"Your previous invalid output:\n{previous_output}\n\n"
            "Please provide a corrected JSON response below:\n"
        )
        return corrected_prompt
