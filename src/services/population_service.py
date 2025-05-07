from typing import Any, Dict, List
from src.services.file_service import FileService
from src.analysis.household_size import compute_household_size_distribution_from_households
from src.repositories.population_repository import PopulationRepository
from src.llm_interface.base_llm import BaseLLM
import pandas as pd

from src.utils.age_bands import assign_broad_age_band, get_broad_age_band_labels

class PopulationService:
    population_repository: PopulationRepository
    file_service: FileService

    def __init__(self):
        self.population_repository = PopulationRepository()
        self.file_service = FileService()
    
    def _compute_statistics(self, households: List[List[Dict[str, Any]]], location: str) -> Dict[str, Any]:
        """Computes household statistics from generated data."""
        return {
            "num_households": len(households),
            "size_distribution": compute_household_size_distribution_from_households(households),
            "synthetic_df": pd.DataFrame([person for household in households for person in household]),
            "census_df": self.file_service.load_age_pyramid(location),
        }
    
    def _format_household_size_phrase(self, sizes: list[int]) -> str:
        if not sizes:
            return ""
        sizes = sorted(sizes)
        if len(sizes) == 1:
            return f"{sizes[0]}"
        elif len(sizes) == 2:
            return f"{sizes[0]} and {sizes[1]}"
        else:
            return f"{', '.join(map(str, sizes[:-1]))} and {sizes[-1]}"


    def _generate_distribution_prompt(self, observed_distribution: dict, location: str, threshold: int = 10, include_stats: bool = True, include_guidance: bool = True) -> str:
        """Generates a structured text prompt based on household size distribution,
        encouraging the LLM to align with real-world distributions rather than enforcing balance.
        """
        target_distribution = self.file_service.load_household_size(location)
        total_obs = sum(observed_distribution.values())
        total_target = sum(target_distribution.values())

        feedback_lines = ["Household Size Distribution (so far):"] if include_stats else []
        increase = []
        decrease = []

        all_sizes = sorted(set(observed_distribution.keys()) | set(target_distribution.keys()))

        for size in all_sizes:
            if size == 0 or size > 6:
                continue

            obs_pct = round((observed_distribution.get(size, 0) / total_obs) * 100)
            tgt_pct = round((target_distribution.get(size, 0) / total_target) * 100)

            label = f"{size}-person"
            if include_stats:
                feedback_lines.append(f"- {label}: {obs_pct}% (target: {tgt_pct}%)")

            diff = obs_pct - tgt_pct
            if abs(diff) >= threshold:
                if diff < 0:
                    increase.append(label)
                else:
                    decrease.append(label)

        suggestions = ["📌 Household Size Guidance:"] if include_guidance else []
        if include_guidance:
            if increase:
                grouped = self._format_household_size_phrase(increase)
                suggestions.append(f"- Increase the number of {grouped} households.")
            if decrease:
                grouped = self._format_household_size_phrase(decrease)
                suggestions.append(f"- Reduce the number of {grouped} households.")
            if not (increase or decrease):
                suggestions.append("- The current distribution is close to target. Continue generating diverse household types.")

        return "\n".join(feedback_lines + [""] + suggestions).strip()

    def _generate_age_distribution_prompt(self, synthetic_df: pd.DataFrame, census_df: pd.DataFrame, threshold: int = 5, include_stats: bool = True, include_guidance: bool = True) -> str:
        # Assign broad bands to synthetic ages
        synthetic_df = synthetic_df.copy()
        synthetic_df["age_band"] = assign_broad_age_band(synthetic_df["age"])
        syn_dist = (
            synthetic_df["age_band"]
            .value_counts(normalize=True)
            .sort_index()
            .multiply(100)
            .round(1)
        )

        # Process census data
        census_df = census_df.copy().reset_index().rename(columns={"age_group": "raw_band"})
        # Extract numeric part of age group to map into bands
        census_df["numeric_age"] = census_df["raw_band"].str.extract(r"(\d+)", expand=False).astype(float)
        census_df["age_band"] = assign_broad_age_band(census_df["numeric_age"])
        census_agg = (
            census_df.groupby("age_band")[["Male", "Female"]]
            .sum()
            .sum(axis=1)
        )
        census_dist = census_agg / census_agg.sum() * 100
        census_dist = census_dist.round(1)

        # Prepare feedback lines
        _, labels = get_broad_age_band_labels()
        lines = ["Age Group Distribution (so far):"] if include_stats else []
        increase, decrease = [], []

        for band in labels:
            syn_pct = syn_dist.get(band, 0)
            tgt_pct = census_dist.get(band, 0)
            diff = syn_pct - tgt_pct
            if include_stats:
                lines.append(f"- {band}: {syn_pct:.1f}% (target: {tgt_pct:.1f}%)")

            if abs(diff) >= threshold:
                if diff < 0:
                    increase.append(band)
                else:
                    decrease.append(band)

        suggestions = ["📌 Age Group Guidance:"] if include_guidance else []
        if include_guidance:
            if increase:
                suggestions.append(f"- Include more individuals in these age groups: {', '.join(increase)}.")
            if decrease:
                suggestions.append(f"- Reduce the number of individuals in these age groups: {', '.join(decrease)}.")
            if not (increase or decrease):
                suggestions.append("- The current age structure is well balanced. Continue maintaining diversity.")

        return "\n".join(lines + [""] + suggestions)


    def _generate_gender_distribution_prompt(self, synthetic_df: pd.DataFrame, threshold: int = 5, include_stats: bool = True, include_guidance: bool = True) -> str:
        gender_counts = synthetic_df["gender"].str.capitalize().value_counts(normalize=True) * 100
        gender_stats = {
            "Male": gender_counts.get("Male", 0),
            "Female": gender_counts.get("Female", 0)
        }
    
        target = {"Male": 50.0, "Female": 50.0}
        lines = ["Gender Distribution (so far):"] if include_stats else []
        suggestions = ["📌 Gender Guidance:"] if include_guidance else []

        for gender in ["Male", "Female"]:
            obs_pct = round(gender_stats.get(gender, 0.0), 1)
            tgt_pct = target[gender]
            diff = obs_pct - tgt_pct
            if include_stats:
                lines.append(f"- {gender}: {obs_pct:.1f}% (target: {tgt_pct:.1f}%)")

            if include_guidance:
                if abs(diff) >= threshold:
                    if diff < 0:
                        suggestions.append(f"- Include more {gender.lower()} individuals.")
                    # else:
                    #     suggestions.append(f"- Fewer {gender.lower()} individuals may be needed.")

        if include_guidance and not suggestions:
            suggestions.append("- The current gender distribution is balanced. Continue maintaining this balance.")

        return "\n".join(lines + [""] + suggestions)



    def _update_prompt_with_statistics(self, base_prompt: str, stats: Dict[str, Any] | None, location: str, include_stats: bool = True, include_guidance: bool = True) -> str:
        """Updates the LLM prompt to incorporate feedback from previous batches."""
        if stats is None:
            return base_prompt.replace("{GUIDANCE}", "").replace("{HOUSEHOLD_STATS}", "").replace("{AGE_STATS}", "").replace("{GENDER_STATS}", "")

        guidance_text = """Feedback from previous households:
The following statistics show the patterns of households generated so far. 
Your task is to improve the realism and diversity of the entire population by generating a household that nudges the distribution toward the targets.
""" if include_stats else ""
        size_stats_text = self._generate_distribution_prompt(stats["size_distribution"], location, 2, include_stats, include_guidance)
        age_stats_text = self._generate_age_distribution_prompt(stats["synthetic_df"], stats["census_df"], 2, include_stats, include_guidance)
        gender_stats_text = self._generate_gender_distribution_prompt(stats["synthetic_df"], 1, include_stats, include_guidance)
        return base_prompt.replace("{GUIDANCE}", guidance_text).replace("{HOUSEHOLD_STATS}", size_stats_text.strip()).replace("{AGE_STATS}", age_stats_text.strip()).replace("{GENDER_STATS}", gender_stats_text.strip())

    def generate_households(self, n_households: int, model: BaseLLM, base_prompt: str, schema: str, batch_size: int, location: str, include_stats: bool, include_guidance: bool) -> list:
        """Generates households in batches using batch processing and feedback-driven prompting."""
        households = []
        prompt = self._update_prompt_with_statistics(base_prompt, None, location, include_stats, include_guidance)

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
                stats = self._compute_statistics(households, location)
                prompt = self._update_prompt_with_statistics(base_prompt, stats, location, include_stats, include_guidance)

        return households
    
    def get_by_id(self, id: str) -> Dict[str, Any]:
        return self.population_repository.get_population_by_id(id)
    
    def save_population(self, population_id: str, households: List[Dict[str, Any]]):
        return self.population_repository.insert_population(population_id, households)
