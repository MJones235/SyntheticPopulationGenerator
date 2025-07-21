import re
from typing import Any, Callable
import pandas as pd

from src.classifiers.household_size.base import HouseholdSizeClassifier
from src.classifiers.household_size.uk_census import UKHouseholdSizeClassifier
from src.classifiers.household_type.base import HouseholdCompositionClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier
from src.services.file_service import FileService
from src.analysis.distributions import (
    compute_age_distribution,
    compute_gender_distribution,
    compute_occupation_distribution,
    compute_target_age_distribution,
)


def generate_distribution_prompt(
    observed_distribution: dict,
    target_distribution: dict,
    label_func: Callable[[Any], str],
    guidance_label: str,
    threshold: int = 10,
    include_stats: bool = True,
    include_guidance: bool = True,
    include_target: bool = True,
) -> str:
    total_obs = sum(observed_distribution.values())
    total_target = sum(target_distribution.values())

    feedback_lines = [f"{guidance_label} Distribution:"] if include_stats else []
    suggestions = [f"{guidance_label} Guidance:"] if include_guidance else []

    increase = []
    decrease = []

    all_keys = sorted(
        set(observed_distribution.keys()) | set(target_distribution.keys())
    )

    for key in all_keys:
        obs_pct = (
            (observed_distribution.get(key, 0) / total_obs) * 100
            if total_obs > 0
            else 0
        )
        tgt_pct = (
            (target_distribution.get(key, 0) / total_target) * 100
            if total_target > 0
            else 0
        )

        if obs_pct == 0.0 and tgt_pct == 0.0:
            continue

        label = label_func(key)
        if include_stats:
            if include_target:
                feedback_lines.append(
                    f"- {label}: current = {obs_pct:.1f}%, target = {tgt_pct:.1f}%"
                )
            else:
                feedback_lines.append(f"- {label}: current = {obs_pct:.1f}%")

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
            suggestions.append(
                f"- The current {guidance_label.lower()} distribution is close to target."
            )

    return "\n".join(
        feedback_lines + [""] + suggestions if feedback_lines else suggestions
    ).strip()

def _build_guidance_text(use_microdata: bool, include_stats: bool, include_target: bool, include_guidance: bool, no_occupation: bool) -> str:
    if include_stats:
        if use_microdata:
            return f"""
The following statistics show the current state of the synthetic population.
Each section displays the distribution of individuals so far, alongside target percentages from Census data.
Your primary task is to preserve the known characteristics of the anchor person and build a plausible household around them.
This includes retaining their age, gender, and other known attributes as given.
Where additional household members must be generated, you should use the census data, where possible, to nudge the overall population toward the target distributions.
Ensure that the household structure remains realistic.
"""
        else:
            if include_target:
                return f"""
The following statistics show the current state of the synthetic population.
Each section shows the current distribution of generated individuals or households, along with the target percentage from Census data.
Your task is to generate a household that nudges the distribution toward the target.
Ensure that the household structure remains realistic.
"""
            else:
                return f"""
The following statistics describe the current distribution of individuals and households in the synthetic population.
Your task is to generate one new household that helps bring this population closer in line with typical population patterns, as reported in Census data.
Use your knowledge of population demographics to identify which values appear over- or underrepresented, and adjust accordingly.
Ensure that the household structure remains realistic.
"""
    elif include_guidance:
        if use_microdata:
            return f"""
The following guidance shows the changes needed to improve the realism and diversity of the entire population, based on Census data.
Your first priority is to preserve the known attributes of the anchor person and construct a plausible household around them.
Where additional household members must be generated, you should use the census data, where possible, to nudge the overall population toward the target distributions.
Ensure that the household structure remains realistic.
"""
        else:
            return f"""
The following guidance shows the changes needed to improve the realism and diversity of the entire population, based on Census data.
Your task is to generate one new household that helps bring this population closer in line with typical population patterns, as reported in Census data.
Ensure that the household structure remains realistic.
"""

    else:
        return ""
    

def update_prompt_with_statistics(
    base_prompt: str,
    synthetic_df: pd.DataFrame | None,
    location: str,
    n_households_generated: int = 0,
    include_stats: bool = True,
    include_guidance: bool = True,
    use_microdata: bool = False,
    include_target: bool = True,
    no_occupation: bool = False,
    no_household_composition: bool = False,
    hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier(),
    hh_size_classifier: HouseholdSizeClassifier = UKHouseholdSizeClassifier()
) -> str:
    """Updates the LLM prompt to incorporate feedback from previous batches."""
    if synthetic_df is None:
        prompt = (
            base_prompt.replace("{N_HOUSEHOLDS}", str(n_households_generated))
            .replace("{GUIDANCE}", "")
            .replace("{HOUSEHOLD_SIZE_STATS}", "")
            .replace("{HOUSEHOLD_COMPOSITION_STATS}", "")
            .replace("{AGE_STATS}", "")
            .replace("{GENDER_STATS}", "")
            .replace("{OCCUPATION_STATS}", "")
        )

        return re.sub(r"\n\s*\n+", "\n\n", prompt).strip()

    guidance_text = _build_guidance_text(use_microdata, include_stats, include_target, include_guidance, no_occupation)

    def build_dist(obs_fn, tgt_fn, label_fn, label, threshold):
        return generate_distribution_prompt(
            observed_distribution=obs_fn(),
            target_distribution=tgt_fn() if include_target or include_guidance else {},
            label_func=label_fn,
            guidance_label=label,
            threshold=threshold,
            include_stats=include_stats,
            include_guidance=include_guidance,
            include_target=include_target,
        )
    
    fs = FileService()

    size_stats_text = build_dist(
        lambda: hh_size_classifier.compute_observed_distribution(synthetic_df),
        lambda: fs.load_household_size(location),
        lambda size: f"{size}-person",
        "Household Size",
        0.5,
    )

    composition_stats_text = ""
    if not no_household_composition:
        composition_stats_text = build_dist(
            lambda: hh_type_classifier.compute_observed_distribution(synthetic_df, "relationship_to_head"),
            lambda: fs.load_household_composition(location),
            lambda composition: composition,
            "Household Composition",
            0.5,
        )

    gender_stats_text = build_dist(
        lambda: compute_gender_distribution(synthetic_df),
        lambda: fs.load_sex_distribution(location),
        lambda gender: gender,
        "Gender",
        0.5,
    )

    age_stats_text = build_dist(
        lambda: compute_age_distribution(synthetic_df),
        lambda: compute_target_age_distribution(fs.load_age_pyramid(location)),
        lambda band: f"{band} years",
        "Age Group",
        1,
    )

    occupation_stats_text = ""
    if not no_occupation:
        occupation_stats_text = build_dist(
            lambda: compute_occupation_distribution(synthetic_df),
            lambda: fs.load_occupation_distribution(location),
            lambda occupation: f"category {occupation}",
            "Occupation",
            0.5,
        )

    prompt = (
        base_prompt.replace("{N_HOUSEHOLDS}", str(n_households_generated))
        .replace("{GUIDANCE}", guidance_text.strip())
        .replace("{HOUSEHOLD_SIZE_STATS}", size_stats_text.strip())
        .replace("{HOUSEHOLD_COMPOSITION_STATS}", composition_stats_text.strip())
        .replace("{AGE_STATS}", age_stats_text.strip())
        .replace("{GENDER_STATS}", gender_stats_text.strip())
        .replace("{OCCUPATION_STATS}", occupation_stats_text.strip())
    ).strip()

    return re.sub(r"\n\s*\n+", "\n\n", prompt).strip()
