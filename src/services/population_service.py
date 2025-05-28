import json
from typing import Any, Dict, List
import random
import pandas as pd
from src.prompts.statistics_feedback import update_prompt_with_statistics
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
        include_target: bool = True
    ) -> List[Dict[str, Any]]:
        households = []
        target_age_distribution = self.file_service.load_age_pyramid(location)

        if use_microdata:
            microdata_df = self.file_service.load_microdata(region)
            sampled_rows = sample_microdata(microdata_df, n_households)
            size_plan = [None] * n_households
        else:
            if compute_household_size:
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
            else:
                size_plan = [None] * n_households

        prompt = update_prompt_with_statistics(
            base_prompt,
            synthetic_df=None,
            target_age_distribution=target_age_distribution,
            location=location,
            n_households_generated=0,
            include_stats=include_stats,
            include_guidance=include_guidance,
            use_microdata=use_microdata,
            include_target=include_target
        )

        for i in range(0, n_households, batch_size):
            batch_count = min(batch_size, n_households - i)
            is_last_batch = (i + batch_count) >= n_households

            print(f"\n--- Generating Batch {i // batch_size + 1} ({batch_count} households) ---")

            batch_prompts = []

            for j in range(batch_count):
                if use_microdata:
                    row = sampled_rows.iloc[i + j]
                    anchor = convert_microdata_row(row)
                    prompt_filled = prompt.replace("{ANCHOR_PERSON}", json.dumps(anchor))
                else:
                    target_size = size_plan[i + j]
                    prompt_filled = prompt.replace(
                        "{NUM_PEOPLE}",
                        str(target_size) + (" person" if target_size == 1 else " people")
                    )
                batch_prompts.append(prompt_filled)

            print(f"Prompt (first in batch): {batch_prompts[0]}")

            try:
                batch_results = model.generate_batch_json(batch_prompts, schema, max_parallel=1, timeout=45)
            except Exception as e:
                print(f"[ERROR] Batch generation failed: {e}")
                batch_results = []

            households.extend(batch_results)

            if not is_last_batch:
                synthetic_df = pd.DataFrame(
                    [dict(**person, household_id=i + 1) for i, household in enumerate(households) for person in household]
                )

                prompt = update_prompt_with_statistics(
                    base_prompt,
                    synthetic_df=synthetic_df,
                    target_age_distribution=target_age_distribution,
                    location=location,
                    n_households_generated=(i + batch_count),
                    include_stats=include_stats,
                    include_guidance=include_guidance,
                    use_microdata=use_microdata,
                    include_target=include_target
                )

        return households

    def get_by_id(self, id: str) -> Dict[str, Any]:
        return self.population_repository.get_population_by_id(id)
    
    def save_population(self, population_id: str, households: List[Dict[str, Any]]):
        return self.population_repository.insert_population(population_id, households)
