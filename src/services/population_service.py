from typing import Any, Dict, List
from src.services.file_service import FileService
from src.analysis.household_size import compute_household_size_distribution_from_households
from src.repositories.population_repository import PopulationRepository
from src.llm_interface.model_factory import LLMFactory
from src.llm_interface.base_llm import BaseLLM
import json

class PopulationService:
    population_repository: PopulationRepository
    file_service: FileService

    def __init__(self):
        self.population_repository = PopulationRepository()
        self.file_service = FileService()
    
    def _compute_statistics(self, households: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Computes household statistics from generated data."""
        return {
            "num_households": len(households),
            "size_distribution": compute_household_size_distribution_from_households(households)
        }
    
    def _generate_distribution_prompt(self, observed_distribution: dict, location: str, threshold: int = 10) -> str:
        """Generates a structured text prompt based on household size distribution,
        encouraging the LLM to align with real-world distributions rather than enforcing balance.
        """
        target_distribution = self.file_service.load_household_size(location)
        total_obs = sum(observed_distribution.values())
        total_target = sum(target_distribution.values())

        feedback_lines = []
        suggestions = []

        all_sizes = sorted(set(observed_distribution.keys()) | set(target_distribution.keys()))

        for size in all_sizes:
            if size == 0:
                continue

            obs_pct = round((observed_distribution.get(size, 0) / total_obs) * 100)
            tgt_pct = round((target_distribution.get(size, 0) / total_target) * 100)

            label = f"{size}-person"
            feedback_lines.append(f"- {label}: {obs_pct}% (target: {tgt_pct}%)")

            diff = obs_pct - tgt_pct
            if abs(diff) >= threshold:
                if diff < 0:
                    suggestions.append(f"- Include more {label} households.")
                else:
                    suggestions.append(f"- Don't generate any more {label} households.")

        if suggestions:
            feedback_lines.append("\nSuggestions:")
            feedback_lines.extend(suggestions)

        return "\n".join(feedback_lines)

        
    def _update_prompt_with_statistics(self, base_prompt: str, stats: Dict[str, Any] | None, location: str) -> str:
        """Updates the LLM prompt to incorporate feedback from previous batches."""
        if stats is None:
            return base_prompt.replace("{HOUSEHOLD_STATS}", "")

        stats_text = f"""
        Statistics for households already generated:
        {self._generate_distribution_prompt(stats["size_distribution"], location, 5)}
        """
        return base_prompt.replace("{HOUSEHOLD_STATS}", stats_text.strip())

    def generate_households(self, n_households: int, model: BaseLLM, base_prompt: str, schema: str, batch_size: int, location: str) -> list:
        """Generates households in batches using batch processing and feedback-driven prompting."""
        households = []
        prompt = self._update_prompt_with_statistics(base_prompt, None, location)

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
                stats = self._compute_statistics(households)
                prompt = self._update_prompt_with_statistics(base_prompt, stats, location)

        return households
    
    def get_by_id(self, id: str) -> Dict[str, Any]:
        return self.population_repository.get_population_by_id(id)
    
    def save_population(self, population_id: str, households: List[Dict[str, Any]]):
        return self.population_repository.insert_population(population_id, households)
