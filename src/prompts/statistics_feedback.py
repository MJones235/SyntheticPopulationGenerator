from typing import Any, Callable, Optional
import pandas as pd

from src.services.file_service import FileService
from src.analysis.distributions import compute_broad_age_distribution, compute_gender_distribution, compute_household_size_distribution, compute_occupation_distribution, compute_target_broad_age_distribution

def generate_distribution_prompt(
    observed_distribution: dict,
    target_distribution: dict,
    label_func: Callable[[Any], str],
    guidance_label: str,
    threshold: int = 10,
    include_stats: bool = True,
    include_guidance: bool = True
) -> str:
    total_obs = sum(observed_distribution.values())
    total_target = sum(target_distribution.values())

    feedback_lines = [f"{guidance_label} Distribution:"] if include_stats else []
    suggestions = [f"{guidance_label} Guidance:"] if include_guidance else []

    increase = []
    decrease = []

    all_keys = sorted(set(observed_distribution.keys()) | set(target_distribution.keys()))

    for key in all_keys:
        obs_pct = (observed_distribution.get(key, 0) / total_obs) * 100 if total_obs > 0 else 0
        tgt_pct = (target_distribution.get(key, 0) / total_target) * 100 if total_target > 0 else 0

        if obs_pct == 0.0 and tgt_pct == 0.0:
            continue

        label = label_func(key)
        if include_stats:
            feedback_lines.append(f"- {label}: current = {obs_pct:.1f}%, target = {tgt_pct:.1f}%")

        diff = obs_pct - tgt_pct
        if include_guidance and abs(diff) >= threshold:
            if diff < 0:
                increase.append(label)
            else:
                decrease.append(label)

    if include_guidance:
        if increase:
            suggestions.append(f"- Increase: {', '.join(increase)}.")
        if decrease:
            suggestions.append(f"- Decrease: {', '.join(decrease)}.")
        if not (increase or decrease):
            suggestions.append(f"- The current {guidance_label.lower()} distribution is close to target.")

    return "\n".join(feedback_lines + [""] + suggestions if feedback_lines else suggestions).strip()


def update_prompt_with_statistics(
    base_prompt: str,
    synthetic_df: pd.DataFrame | None,
    target_age_distribution: pd.DataFrame,
    location: str,
    n_households_generated: int = 0,
    include_stats: bool = True,
    include_guidance: bool = True,
    use_microdata: bool = False) -> str:
    """Updates the LLM prompt to incorporate feedback from previous batches."""
    if synthetic_df is None:
        return (
            base_prompt
            .replace("{N_HOUSEHOLDS}", str(n_households_generated))
            .replace("{GUIDANCE}", "")
            .replace("{HOUSEHOLD_STATS}", "")
            .replace("{AGE_STATS}", "")
            .replace("{GENDER_STATS}", "")
            .replace("{OCCUPATION_STATS}", "")
        ).strip()

    if include_stats:
        if use_microdata:
            guidance_text = """
The following statistics show the current state of the synthetic population.
Each section displays the distribution of individuals or households so far, alongside target percentages from the 2021 Census.
Your primary task is to preserve the known characteristics of the anchor person and build a plausible household around them.
This includes retaining their age, gender, and other known attributes as given.
Where additional household members must be generated, you should use the census data, where possible, to:
- Nudge the overall population toward the target distributions.
- Select a household size that is currently underrepresented.
- Include individuals from underrepresented age groups.
- Include underrepresented occupations.
- Maintain an even gender balance across the full population.
All household structures must remain realistic and demographically plausible.
"""
        else:
            guidance_text = """
The following statistics show the current state of the synthetic population.
Each section shows the current distribution of generated individuals or households, along with the target percentage from Census 2021 data.
Your task is to generate a household that nudges the distribution toward the target.
Select a household size that is currently underrepresented.
If possible, include individuals from underrepresented age groups. 
Include underrepresented occupations. 
Maintain an even gender balance across the full population.
Ensure that the household structure remains realistic.
"""
    elif include_guidance:
        if use_microdata:
            guidance_text = """
The following guidance shows the changes needed to improve the realism and diversity of the entire population, based on Census 2021 data.
Your first priority is to preserve the known attributes of the anchor person and construct a plausible household around them.
Where additional household members must be generated, you should use the census data, where possible, to:
- Nudge the overall population toward the target distributions.
- Select a household size that is currently underrepresented.
- Include individuals from underrepresented age groups.
- Include underrepresented occupations.
- Maintain an even gender balance across the full population.
All household structures must remain realistic and demographically plausible.
"""
        else:
            guidance_text = """
The following guidance shows the changes needed to improve the realism and diversity of the entire population, based on Census 2021 data.
Select a household size that is currently underrepresented.
If possible, include individuals from underrepresented age groups.  
Include underrepresented occupations.
Maintain an even gender balance across the full population.
Ensure that the household structure remains realistic.
"""

    else:
        guidance_text = ""

    # Household Size Distribution
    observed_size_dist = compute_household_size_distribution(synthetic_df)
    target_size_dist = FileService().load_household_size(location)

    def household_label(size):
        return f"{size}-person"

    size_stats_text = generate_distribution_prompt(
        observed_distribution=observed_size_dist,
        target_distribution=target_size_dist,
        label_func=household_label,
        guidance_label="Household Size",
        threshold=0.5,
        include_stats=include_stats,
        include_guidance=include_guidance
    )

    # Age Distribution
    syn_age_dist = compute_broad_age_distribution(synthetic_df)
    target_age_dist = compute_target_broad_age_distribution(target_age_distribution)


    def age_label(band):
        return f"{band} years"


    age_stats_text = generate_distribution_prompt(
        observed_distribution=syn_age_dist,
        target_distribution=target_age_dist,
        label_func=age_label,
        guidance_label="Age Group",
        threshold=1,
        include_stats=include_stats,
        include_guidance=include_guidance
    )

    # Gender Distribution
    gender_dist = compute_gender_distribution(synthetic_df)

    def gender_label(gender):
        return gender

    gender_stats_text = generate_distribution_prompt(
        observed_distribution=gender_dist,
        target_distribution={"Male": 50.0, "Female": 50.0},
        label_func=gender_label,
        guidance_label="Gender",
        threshold=0.5,
        include_stats=include_stats,
        include_guidance=include_guidance
    )

    # Occupation Distribution
    observed_occupation_dist = compute_occupation_distribution(synthetic_df)

    target_occupation_dist = FileService().load_occupation_distribution(location)

    def occupation_label(occupation):
        return f"cateogry {occupation}"

    occupation_stats_text = generate_distribution_prompt(
        observed_distribution=observed_occupation_dist,
        target_distribution=target_occupation_dist,
        label_func=occupation_label,
        guidance_label="Occupation",
        threshold=0.5,
        include_stats=include_stats,
        include_guidance=include_guidance
    )

    return (
        base_prompt
        .replace("{N_HOUSEHOLDS}", str(n_households_generated))
        .replace("{GUIDANCE}", guidance_text.strip())
        .replace("{HOUSEHOLD_STATS}", size_stats_text.strip())
        .replace("{AGE_STATS}", age_stats_text.strip())
        .replace("{GENDER_STATS}", gender_stats_text.strip())
        .replace("{OCCUPATION_STATS}", occupation_stats_text.strip())
    ).strip()
