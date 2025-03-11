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

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Subclasses must implement how to call the LLM and return a raw text response.
        """
        raise NotImplementedError
    
    @abstractmethod
    def generate_batch_text(self, prompt: list[str]) -> list[str]:
        """
        Subclasses must implement how to call the LLM and return a raw text response.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_model_metadata(self) -> str:
        raise NotImplementedError

    def generate_household(
        self,
        prompt: str,
        json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Uses generate_text(prompt) to obtain a JSON array describing households,
        then validates it against the provided json_schema. 
        In case of failure to parse or validate, optionally re-prompts the model
        up to max_retries times.

        :param prompt: The initial prompt instructing the model to return JSON data
        :param json_schema: A jsonschema dict to validate the structure
        :param max_retries: How many times we attempt to repair the response
        :param correction_prompt_template: Template for re-prompting if invalid
        :return: A list of household dicts, or an empty list if all retries fail
        """
        attempts = 0
        current_prompt = prompt

        while attempts < self.max_retries:
            try:
                raw_response = self.generate_text(current_prompt).strip()
            except Exception as e:
                attempts += 1
                print(f"Failed to generate response: {str(e)}.  Retrying...")
                continue

            # Try to parse as JSON
            try:
                data = json.loads(raw_response)
            except json.JSONDecodeError as e:
                # If we fail to parse JSON, we can attempt to re-prompt
                attempts += 1
                error_msg = f"JSON parse error: {str(e)}"
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    previous_output=raw_response,
                    error=error_msg,
                    schema=json_schema,
                )
                print("Unable to parse response.  Retrying...")
                continue  # Retry with corrected prompt

            # If JSON parse succeeded, we now validate the structure
            try:
                jsonschema.validate(instance=data, schema=json_schema)
                # If no exception, we have valid data according to the schema
                return data
            except jsonschema.ValidationError as e:
                # The data structure didn't match the schema
                attempts += 1
                error_msg = f"Schema validation error: {str(e)}"
                current_prompt = self._build_correction_prompt(
                    original_prompt=prompt,
                    previous_output=raw_response,
                    error=error_msg,
                    schema=json_schema,
                )
                print("Invalid JSON schema.  Retrying...")

        # If we exit the loop, all retries failed
        return []

    def generate_batch_households(self, prompts: List[str], json_schema: Dict[str, Any], max_parallel=1) -> List[Dict[str, Any]]:
        """Generates multiple households in parallel using batch processing, schema validation, and controlled parallelism.
        
        Args:
            prompts (List[str]): List of prompts instructing the model to return JSON data.
            json_schema (Dict[str, Any]): A jsonschema dict to validate the structure.
            max_parallel (int): Maximum number of households processed in parallel.

        Returns:
            tuple: (valid_households, failed_prompts)
                - valid_households (List[Dict[str, Any]]): List of valid household dictionaries.
                - failed_prompts (List[str]): List of prompts that failed after all retries.
        """
        failed_prompts = list(prompts)
        valid_households = []

        for attempt in range(self.max_retries):
            if not failed_prompts:
                break  # Exit if no failures

            new_failed_prompts = []
            batch_start = time.time()

            for i in range(0, len(failed_prompts), max_parallel):
                print(f"[INFO] {"Generating" if attempt == 0 else "Regenerating"} households {i+1} to {i+max_parallel}")
                batch_prompts = failed_prompts[i : i + max_parallel]  # Process in chunks
                try:
                    batch_responses = self.generate_batch_text(batch_prompts)
                except TimeoutError as e:
                    print(f"[ERROR] LLM batch request timed out: {e}")
                    new_failed_prompts.extend(batch_prompts)  # Retry entire batch later
                    continue

                for prompt, response in zip(batch_prompts, batch_responses):
                    if response is None:
                        print(f"[WARNING] Missing response. Retrying...")
                        new_failed_prompts.append(prompt)
                        continue

                    try:
                        data = json.loads(response)
                        jsonschema.validate(instance=data, schema=json_schema)
                        valid_households.append(data["household"])  # Extract and extend
                    except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                        print(f"[ERROR] Response validation failed. Retrying...")
                        new_failed_prompts.append(self._build_correction_prompt(prompt, response, f"Validation error: {e}", json_schema))

            failed_prompts = new_failed_prompts
            batch_end = time.time()
            print(f"[INFO] Batch completed in {batch_end - batch_start:.2f} seconds.\n\n")

        return valid_households


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
