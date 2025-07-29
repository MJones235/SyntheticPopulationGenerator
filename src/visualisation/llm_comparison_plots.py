import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.classifiers.household_size.dar_es_salaam import DarEsSalaamHouseholdSizeClassifier
from src.classifiers.household_size.un_global import UNHouseholdSizeClassifier
from src.classifiers.household_type.un_global import UNHouseholdCompositionClassifier
from src.analysis.similarity_metrics import get_census_age_pyramid, get_synthetic_age_pyramid
from src.services.file_service import FileService
from src.classifiers.household_size.uk_census import UKHouseholdSizeClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier
from src.services.population_service import PopulationService

location = "Dar es Salaam"
experiement_name = f"{location.replace(" ", "")}_method_b"
hh_size_classifier = DarEsSalaamHouseholdSizeClassifier()
hh_type_classifier = UNHouseholdCompositionClassifier()
population_service = PopulationService()
file_service = FileService()

# Newcaste Method B
#LLM_MODEL_TO_POPULATION_ID = {
#    "llama3.1:8b": "baa6c346-3e6d-47c1-a498-4f6a9fbf1b0c",
#    "gpt-4o-mini": "027da695-f13b-48f0-b69e-a708eec61457",
#    "gpt-4o": "ecc1527e-cd51-4d22-98d2-7aee56b80ec5",
#    "o3-mini": "72292b2f-cc14-450e-9f22-df18bbd8cfa4",
#    "DeepSeek-R1-0528": "bb02c3f8-4044-45f2-a43b-c939f246d076",
#}


# Newcastle Method C
#LLM_MODEL_TO_POPULATION_ID = {
#    "gpt-4o": "04794b00-8fa5-4308-9414-e477b49ee18d",
#    "DeepSeek-R1-0528": "c09bf2ce-fc0b-4d0e-ae9c-20e6d19462c2",
#}

# Dar es Salaam Method B
LLM_MODEL_TO_POPULATION_ID = {
    "gpt-4o": "8052f98d-8a0f-4b38-b6c2-b310138eb8b4",
    "DeepSeek-R1-0528": "2f6df1ba-c312-4a8b-b278-998221d38efa",
}

# Dar es Salaam Method C
#LLM_MODEL_TO_POPULATION_ID = {
#    "gpt-4o": "129a2f40-2c94-4e55-91ed-2af141b06f8c",
#    "DeepSeek-R1-0528": "93e307e9-6727-4228-bba3-2c04b895ea06",
#}

def plot_categorical_distributions(
    distributions: dict[str, dict],
    xlabel: str,
    output_name: str,
    census_distribution: dict = None,
    label_order: list[str] = None,
):
    os.makedirs("outputs/plots", exist_ok=True)

    # Style setup
    sns.set(style="whitegrid")
    total_series = len(distributions) + int(census_distribution is not None)
    colors = sns.color_palette("colorblind", n_colors=total_series)

    # Category ordering
    all_keys = set().union(*(d.keys() for d in distributions.values()))
    if census_distribution:
        all_keys = all_keys.union(census_distribution.keys())

    if label_order:
        categories = [label for label in label_order if label in all_keys]
    else:
        categories = sorted(all_keys)

    x = np.arange(len(categories))
    bar_width = 0.8 / total_series

    fig, ax = plt.subplots(figsize=(max(10, len(categories)), 5))

    is_numeric_labels = all(str(k).isdigit() for k in categories)
    rotation = 0 if is_numeric_labels else 45
    wrapped_categories = categories if is_numeric_labels else [label.replace("with ", "with\n ").replace(" aged", "\naged") for label in categories]



    # Plot synthetic distributions
    for i, (label, dist) in enumerate(distributions.items()):
        y_vals = [dist.get(k, 0) for k in categories]
        ax.bar(x + i * bar_width, y_vals, width=bar_width, label=label, color=colors[i])

    # Plot census distribution
    if census_distribution:
        y_vals = [census_distribution.get(k, 0) for k in categories]
        i = len(distributions)
        ax.bar(x + i * bar_width, y_vals, width=bar_width, label="Census", color="grey")


    # Axes and legend
    ax.set_xticks(x + bar_width * (total_series - 1) / 2)
    ax.set_xticklabels(wrapped_categories, rotation=rotation, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage of Households (%)")

    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)

    # Save
    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(f"outputs/plots/{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multi_age_pyramid(synthetic_dfs: dict[str, pd.DataFrame], census_df: pd.DataFrame, output_name: str):
    sns.set(style="whitegrid")
    os.makedirs("outputs/plots", exist_ok=True)

    model_names = list(synthetic_dfs.keys())
    n_models = len(model_names)
    palette = sns.color_palette("colorblind", n_colors=n_models)

    census_pyr = get_census_age_pyramid(census_df)
    age_groups = census_pyr.index.tolist()
    bar_height = 0.8 / (n_models + 1.5)
    y_positions = np.arange(len(age_groups))

    fig, ax = plt.subplots(figsize=(12, 6))


    max_val = max(
        abs(census_pyr.values).max(),
        max(abs(get_synthetic_age_pyramid(df).values).max() for df in synthetic_dfs.values())
    )
    x_margin = 1.0

    # Background shading
    ax.axvspan(-max_val - x_margin, 0, color="lightblue", alpha=0.5, zorder=0)
    ax.axvspan(0, max_val + x_margin, color="mistyrose", alpha=0.5, zorder=0)

    # Plot synthetic models
    for i, (model, df) in enumerate(synthetic_dfs.items()):
        syn_pyr = get_synthetic_age_pyramid(df).reindex(age_groups).fillna(0)
        offset = (i - n_models / 2) * bar_height

        ax.barh(
            y=y_positions + offset,
            width=-syn_pyr["Male"],
            height=bar_height,
            color=palette[i],
            label=model,
            edgecolor="black"
        )
        ax.barh(
            y=y_positions + offset,
            width=syn_pyr["Female"],
            height=bar_height,
            color=palette[i],
            edgecolor="black"
        )

    # Census bars (gray, rightmost)
    census_offset = (n_models - 1) * bar_height

    ax.barh(
        y=y_positions + census_offset,
        width=-census_pyr["Male"],
        height=bar_height,
        color="grey",
        label="Census",
        edgecolor="black"
    )
    ax.barh(
        y=y_positions + census_offset,
        width=census_pyr["Female"],
        height=bar_height,
        color="grey",
        edgecolor="black"
    )

    # Vertical midline
    ax.axvline(0, color="black", lw=1)

    # Axis settings
    ax.set_yticks(y_positions)
    ax.set_yticklabels(age_groups)
    
    ax.set_xlim(-max_val - x_margin, max_val + x_margin)


    ax.set_xlabel("Percentage of Population (%)")
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    # Side labels
    ax.text(-max_val / 2, y_positions[-1] - 0.5, "Male", fontsize=11, ha="left", va="bottom", color="black", fontweight="bold")
    ax.text(max_val / 2, y_positions[-1] - 0.5, "Female", fontsize=11, ha="right", va="bottom", color="black", fontweight="bold")

    # Legend: one entry per model
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0, fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(f"outputs/plots/{output_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)




hh_size_synthetic_distributions = {}
hh_composition_synthetic_distributions = {}
dfs = {}

for model, pop_id in LLM_MODEL_TO_POPULATION_ID.items():
    df = pd.DataFrame(population_service.get_by_id(pop_id))
    hh_size_synthetic_distributions[model] = hh_size_classifier.compute_observed_distribution(df)
    hh_composition_synthetic_distributions[model] = hh_type_classifier.compute_observed_distribution(df)
    dfs[model] = df

# Optionally load census distribution
hh_size_census_distribution = file_service.load_household_size(location)
hh_composition_census_distribution = file_service.load_household_composition(location)

plot_categorical_distributions(
    distributions=hh_size_synthetic_distributions,
    xlabel="Household Size",
    output_name=f"{experiement_name}_hh_size",
    census_distribution=hh_size_census_distribution
)

plot_categorical_distributions(
    distributions=hh_composition_synthetic_distributions,
    xlabel="Household Type",
    output_name=f"{experiement_name}_hh_composition",
    census_distribution=hh_composition_census_distribution,
    label_order=hh_type_classifier.get_label_order()
)

plot_multi_age_pyramid(
    synthetic_dfs=dfs,
    census_df=file_service.load_age_pyramid(location),
    output_name=f"{experiement_name}_age_pyramid"
)
