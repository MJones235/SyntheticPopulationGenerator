from abc import ABC, abstractmethod
from typing import Any, Dict, List
import json
import jsonschema

class BaseLLM(ABC):
    """
    An abstract base class defining the standard interface
    for local or API-based LLMs.
    It includes logic for generating JSON-based outputs,
    validating them, and retrying if necessary.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    @abstractmethod
    def generate_text(self, prompt: str) -> str:
        """
        Subclasses must implement how to call the LLM and return a raw text response.
        """
        raise NotImplementedError

    def generate_households(
        self,
        prompt: str,
        json_schema: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
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
            raw_response = self.generate_text(current_prompt).strip()

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

        # If we exit the loop, all retries failed
        return []

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
