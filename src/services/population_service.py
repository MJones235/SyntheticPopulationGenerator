import concurrent
from typing import Any, Dict, List
from src.analysis.household_size import compute_household_size_distribution_from_households
from src.repositories.population_repository import PopulationRepository
from src.llm_interface.model_factory import LLMFactory
from src.llm_interface.base_llm import BaseLLM
import json

class PopulationService:
    population_repository: PopulationRepository

    def __init__(self):
        self.population_repository = PopulationRepository()

    def generate_single_household(self, n: int, n_total: int, model_type:str, model_kwargs: dict, prompt: str, schema: str):
        print(f"Generating household {n + 1}/{n_total}")
        try:
            model = LLMFactory.get_provider(model_type, **model_kwargs)
            result= model.generate_json(prompt, schema)
            return result["household"]
        except Exception as e:
            print("[ERROR] Error generating household. Skipping...")
            print(e)
            return None
    
    def _compute_statistics(self, households: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Computes household statistics from generated data."""
        return {
            "num_households": len(households),
            "size_distribution": compute_household_size_distribution_from_households(households)
        }
    
    def _generate_distribution_prompt(self, size_distribution: dict) -> str:
        """Generates a structured text prompt based on household size distribution,
        encouraging the LLM to align with real-world distributions rather than enforcing balance.
        """
        
        sorted_sizes = sorted(size_distribution.items(), key=lambda x: x[1], reverse=True)
        most_common = [f"{size}-person ({round(percent)}%)" for size, percent in sorted_sizes if percent > 0]
        missing_sizes = [str(size) for size, percent in size_distribution.items() if percent == 0]
        prompt_parts = []
        
        if most_common:
            prompt_parts.append(f"Current household size distribution: {', '.join(most_common)}.")

        if missing_sizes:
            prompt_parts.append(f"No {', '.join(missing_sizes)}-person households have been generated yet.")

        prompt_parts.append("Adjust future households to better reflect typical distributions for this location. "
                            "Use your general knowledge to align household sizes with what is realistic in this area.")

        return " ".join(prompt_parts)

        
    def _update_prompt_with_statistics(self, base_prompt: str, stats: Dict[str, Any] | None) -> str:
        """Updates the LLM prompt to incorporate feedback from previous batches."""
        if stats is None:
            return base_prompt.replace("{HOUSEHOLD_STATS}", "")

        stats_text = f"""
        Statistics for households already generated:
        {self._generate_distribution_prompt(stats["size_distribution"])}
        """
        return base_prompt.replace("{HOUSEHOLD_STATS}", stats_text.strip())

    def generate_households(self, n_households: int, model: BaseLLM, base_prompt: str, schema: str, batch_size: int) -> list:
        """Generates households in batches using batch processing and feedback-driven prompting."""
        households = []
        prompt = self._update_prompt_with_statistics(base_prompt, None)

        for i in range(0, n_households, batch_size):
            batch_count = min(batch_size, n_households - i)
            is_last_batch = (i + batch_count) >= n_households

            print(f"\n--- Generating Batch {i // batch_size + 1} ({batch_count} households) ---")

            try:
                batch_prompts = [prompt] * batch_count
                batch_results = model.generate_batch_json(batch_prompts, schema)
            except Exception as e:
                print(f"[ERROR] Batch generation failed: {e}")
                batch_results = []

            households.extend(batch_results)

            if not is_last_batch:
                stats = self._compute_statistics(households)
                prompt = self._update_prompt_with_statistics(base_prompt, stats)
                print(f"Updated Statistics after Batch {i // batch_size + 1}: {json.dumps(stats, indent=2)}")

        return households
    
    def get_by_id(self, id: str) -> Dict[str, Any]:
        return self.population_repository.get_population_by_id(id)
    
    def save_population(self, population_id: str, households: List[Dict[str, Any]]):
        return self.population_repository.insert_population(population_id, households)
