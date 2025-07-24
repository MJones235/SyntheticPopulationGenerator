from typing import Any, Dict, List, Optional
import random
import pandas as pd
from src.classifiers.household_size.base import HouseholdSizeClassifier
from src.classifiers.household_size.uk_census import UKHouseholdSizeClassifier
from src.classifiers.household_type.base import HouseholdCompositionClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier
from src.prompts.statistics_feedback import update_prompt_with_statistics as prepare_prompt
from src.services.file_service import FileService
from src.repositories.population_repository import PopulationRepository
from src.llm_interface.base_llm import BaseLLM
from src.utils.microdata_decoder import convert_microdata_row
from src.utils.microdata_sampler import sample_microdata

class PopulationService:
    population_repository: PopulationRepository
    file_service: FileService

    def __init__(self):
        self.population_repository = PopulationRepository()
        self.file_service = FileService()

    def generate_households(
        self,
        n_households: int,
        model: BaseLLM,
        base_prompt: str,
        schema: str,
        location: str,
        region: str,
        batch_size: int,
        include_stats: bool,
        include_guidance: bool,
        use_microdata: bool = False,
        compute_household_size: bool = False,
        include_target: bool = True,
        no_occupation: bool = False,
        n_run: int = 1,
        no_household_composition: bool = False,
        include_avg_household_size: bool = False,
        custom_guidance: Optional[str] = None,
        hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier(),
        hh_size_classifier: HouseholdSizeClassifier = UKHouseholdSizeClassifier()
    ) -> List[Dict[str, Any]]:
        
        households = []
        size_plan = self._plan_household_sizes(n_households, location) if compute_household_size else [None] * n_households

        if use_microdata:
            microdata_df = self.file_service.load_microdata(region)
            sampled_rows = sample_microdata(microdata_df, n_households)
        
        prompt = prepare_prompt(
            base_prompt,
            synthetic_df=None,
            location=location,
            n_households_generated=0,
            include_stats=include_stats,
            include_guidance=include_guidance,
            use_microdata=use_microdata,
            include_target=include_target,
            no_occupation=no_occupation,
            no_household_composition=no_household_composition,
            include_avg_household_size=include_avg_household_size,
            custom_guidance=custom_guidance,
            hh_type_classifier=hh_type_classifier,
            hh_size_classifier=hh_size_classifier
        )

        for i in range(0, n_households, batch_size):
            batch_count = min(batch_size, n_households - i)
            is_last_batch = (i + batch_count) >= n_households

            print(f"\n--- Generating Batch {i // batch_size + 1} ({batch_count} households), Run {n_run} ---")

            batch_prompts = self._prepare_batch_prompts(
                prompt,
                size_plan[i:i+batch_count],
                sampled_rows.iloc[i:i+batch_count] if use_microdata else None
            )

            print(f"Prompt (first in batch): {batch_prompts[0]}")

            batch_results = self._run_batch(model, batch_prompts, schema)
            households.extend(batch_results)

            if not is_last_batch:
                synthetic_df = pd.DataFrame(
                    [dict(**person, household_id=i + 1) for i, household in enumerate(households) for person in household]
                )

                prompt = prepare_prompt(
                    base_prompt,
                    synthetic_df=synthetic_df,
                    location=location,
                    n_households_generated=(i + batch_count),
                    include_stats=include_stats,
                    include_guidance=include_guidance,
                    use_microdata=use_microdata,
                    include_target=include_target,
                    no_occupation=no_occupation,
                    no_household_composition=no_household_composition,
                    include_avg_household_size=include_avg_household_size,
                    custom_guidance=custom_guidance,
                    hh_type_classifier=hh_type_classifier,
                    hh_size_classifier=hh_size_classifier
                )

        return households
    
    def _plan_household_sizes(self, n_households: int, location: str) -> List[Optional[int]]:
        size_distribution = self.file_service.load_household_size(location)
        total = sum(size_distribution.values())
        size_distribution = {k: v / total for k, v in size_distribution.items()}
        size_counts = {
            size: int(round(n_households * proportion))
            for size, proportion in size_distribution.items()
        }

        size_plan = []
        for size, count in size_counts.items():
            size_plan.extend([size] * count)

        random.shuffle(size_plan)
        while len(size_plan) < n_households:
            size_plan.append(random.choice(list(size_distribution.keys())))
        while len(size_plan) > n_households:
            size_plan.pop()

        return size_plan
    
    def _prepare_batch_prompts(self, prompt_template: str, size_plan: List[Optional[int]], sampled_rows: Optional[pd.DataFrame]) -> List[str]:
        batch_prompts = []
        for i, target_size in enumerate(size_plan):
            if sampled_rows is not None:
                row = sampled_rows.iloc[i]
                anchor = convert_microdata_row(row)
                prompt_filled = prompt_template.replace("{ANCHOR_PERSON}", anchor)
            else:
                prompt_filled = prompt_template.replace(
                    "{NUM_PEOPLE}",
                    str(target_size) + (" person" if target_size == 1 else " people")
                )
            batch_prompts.append(prompt_filled)
        return batch_prompts
    
    def _run_batch(self, model: BaseLLM, prompts: List[str], schema: str) -> List[Dict[str, Any]]:
        try:
            return model.generate_batch_json(prompts, schema, max_parallel=1, timeout=45)
        except Exception as e:
            print(f"[ERROR] Batch generation failed: {e}")
            return []

    def get_by_id(self, id: str) -> Dict[str, Any]:
        return self.population_repository.get_population_by_id(id)
    
    def save_population(self, population_id: str, households: List[Dict[str, Any]]):
        return self.population_repository.insert_population(population_id, households)
