from typing import Any, Dict, List

import pandas as pd
from src.prompts.statistics_feedback import update_prompt_with_statistics
from src.services.file_service import FileService
from src.repositories.population_repository import PopulationRepository
from src.llm_interface.base_llm import BaseLLM

class PopulationService:
    population_repository: PopulationRepository
    file_service: FileService

    def __init__(self):
        self.population_repository = PopulationRepository()
        self.file_service = FileService()

    def generate_households(self, n_households: int, model: BaseLLM, base_prompt: str, schema: str, batch_size: int, location: str, include_stats: bool, include_guidance: bool) -> list:
        """Generates households in batches using batch processing and feedback-driven prompting."""
        households = []
        target_age_distribution = self.file_service.load_age_pyramid(location)
        prompt = update_prompt_with_statistics(base_prompt, None, target_age_distribution, location, include_stats, include_guidance)

        for i in range(0, n_households, batch_size):
            batch_count = min(batch_size, n_households - i)
            is_last_batch = (i + batch_count) >= n_households

            print(f"\n--- Generating Batch {i // batch_size + 1} ({batch_count} households) ---")

            print(f"Prompt: {prompt}")

            try:
                batch_prompts = [prompt] * batch_count
                batch_results = model.generate_batch_json(batch_prompts, schema, max_parallel=1, timeout=45)
            except Exception as e:
                print(f"[ERROR] Batch generation failed: {e}")
                batch_results = []

            households.extend(batch_results)

            if not is_last_batch:
                synthetic_df = pd.DataFrame(
                    [dict(**person, household_id=i + 1) for i, household in enumerate(households) for person in household]
                )

                prompt = update_prompt_with_statistics(base_prompt, synthetic_df, target_age_distribution, location, include_stats, include_guidance)

        return households
    
    def get_by_id(self, id: str) -> Dict[str, Any]:
        return self.population_repository.get_population_by_id(id)
    
    def save_population(self, population_id: str, households: List[Dict[str, Any]]):
        return self.population_repository.insert_population(population_id, households)
