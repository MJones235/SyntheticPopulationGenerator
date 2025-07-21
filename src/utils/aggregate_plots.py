import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.classifiers.household_type.base import HouseholdCompositionClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier

def plot_household_size_aggregate(synthetic_runs: list[dict], census: dict):
    # Collect all keys
    all_keys = sorted(set().union(*[set(d.keys()) for d in synthetic_runs], set(census.keys())))

    # Normalize and align
    data = []
    for run in synthetic_runs:
        normed = {k: run.get(k, 0.0) for k in all_keys}
        data.append([normed[k] for k in all_keys])

    data = np.array(data)
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    n = len(data)
    ci = 1.96 * (std / np.sqrt(n))

    census_vals = [census.get(k, 0.0) for k in all_keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    indices = np.arange(len(all_keys))
    bar_width = 0.35

    ax.bar(indices - bar_width, census_vals, width=bar_width, label="Census", color="blue", alpha=0.6)
    ax.bar(indices, mean, width=bar_width, yerr=ci, label="Synthetic (mean ± CI)", color="orange", alpha=0.6, capsize=3)

    ax.set_xticks(indices)
    ax.set_xticklabels(all_keys)
    ax.set_xlabel("Household Size")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Aggregate Household Size Distribution")
    ax.legend()
    return fig

def plot_age_pyramid_aggregate(synthetic_dfs: list[pd.DataFrame], census_df: pd.DataFrame):
    from src.analysis.similarity_metrics import get_synthetic_age_pyramid, get_census_age_pyramid

    # Get all age pyramids (percent distributions) for each run
    male_dists = []
    female_dists = []
    for df in synthetic_dfs:
        pct = get_synthetic_age_pyramid(df)
        male_dists.append(pct.get("Male", pd.Series(0, index=pct.index)))
        female_dists.append(pct.get("Female", pd.Series(0, index=pct.index)))

    male_matrix = pd.concat(male_dists, axis=1).fillna(0)
    female_matrix = pd.concat(female_dists, axis=1).fillna(0)

    male_mean = male_matrix.mean(axis=1)
    female_mean = female_matrix.mean(axis=1)

    male_ci = 1.96 * male_matrix.std(axis=1) / np.sqrt(male_matrix.shape[1])
    female_ci = 1.96 * female_matrix.std(axis=1) / np.sqrt(female_matrix.shape[1])

    # Census reference
    census_pct = get_census_age_pyramid(census_df)
    census_male = census_pct.get("Male", pd.Series(0, index=male_mean.index))
    census_female = census_pct.get("Female", pd.Series(0, index=female_mean.index))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Synthetic bars with CI
    ax.barh(
        male_mean.index,
        -male_mean,
        xerr=male_ci,
        color="blue",
        alpha=0.6,
        label="Synthetic Male (mean ± 95% CI)",
        capsize=3
    )
    ax.barh(
        female_mean.index,
        female_mean,
        xerr=female_ci,
        color="red",
        alpha=0.6,
        label="Synthetic Female (mean ± 95% CI)",
        capsize=3
    )

    # Census shaded
    ax.barh(
        census_male.index,
        -census_male,
        color="lightblue",
        alpha=0.4,
        edgecolor="black",
        label="Census Male",
    )
    ax.barh(
        census_female.index,
        census_female,
        color="pink",
        alpha=0.4,
        edgecolor="black",
        label="Census Female",
    )

    ax.axvline(0, color="black")
    ax.set_xlabel("Percentage of Population")
    ax.set_ylabel("Age Group")
    ax.set_title("Aggregate Age Pyramid: Synthetic vs Census")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

def plot_occupations_aggregate(synthetic_dicts: list[dict], census: dict):
    # Union of all keys across all runs and census
    all_categories = sorted(set().union(*[d.keys() for d in synthetic_dicts], census.keys()))
    
    # Normalize and align each run
    aligned_data = []
    for d in synthetic_dicts:
        row = [d.get(cat, 0.0) for cat in all_categories]
        total = sum(row)
        if total > 0:
            row = [(v * 100 / total) for v in row]  # ensure percentages
        aligned_data.append(row)

    data = np.array(aligned_data)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    ci = 1.96 * stds / np.sqrt(data.shape[0])

    # Census (align and ensure %)
    census_row = [census.get(cat, 0.0) for cat in all_categories]
    total_census = sum(census_row)
    census_pct = [(v * 100 / total_census) if total_census > 0 else 0.0 for v in census_row]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    indices = np.arange(len(all_categories))
    bar_width = 0.35

    ax.bar(
        indices - bar_width,
        census_pct,
        width=bar_width,
        label="Census",
        color="blue",
        alpha=0.6
    )
    ax.bar(
        indices,
        means,
        width=bar_width,
        yerr=ci,
        label="Synthetic (mean ± 95% CI)",
        color="orange",
        alpha=0.6,
        capsize=3
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(all_categories, rotation=45, ha="right")
    ax.set_xlabel("Standard Occupation Category")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Aggregate Occupation Distribution")
    ax.legend()
    ax.grid(axis="y")

    fig.tight_layout()
    return fig


def plot_household_structure_bar_aggregate(
    dfs: list[pd.DataFrame], census_df: pd.DataFrame, hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier()
) -> plt.Figure:
    label_order = hh_type_classifier.get_label_order()
    # Get synthetic composition for each run
    run_distributions = []
    for df in dfs:
        try:
            dist = hh_type_classifier.compute_observed_distribution(df)
            run_distributions.append(dist)
        except Exception:
            continue

    # Align all distributions to the same label set
    aligned = []
    for dist in run_distributions:
        row = [dist.get(label, 0.0) for label in label_order]
        total = sum(row)
        if total > 0:
            row = [(v * 100 / total) for v in row]
        aligned.append(row)

    data = np.array(aligned)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    ci = 1.96 * stds / np.sqrt(data.shape[0])

    # Census composition
    census_counts = census_df
    census_pct = [census_counts.get(label, 0.0) for label in label_order]
    total_census = sum(census_pct)
    if total_census > 0:
        census_pct = [(v * 100 / total_census) for v in census_pct]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(label_order))
    bar_width = 0.35

    ax.bar(
        x - bar_width,
        census_pct,
        width=bar_width,
        label="Census",
        color="royalblue",
    )
    ax.bar(
        x,
        means,
        width=bar_width,
        yerr=ci,
        label="Synthetic (mean ± 95% CI)",
        color="orange",
        alpha=0.8,
        capsize=3
    )

    ax.set_xticks(x)
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    ax.set_ylabel("Percentage of Households")
    ax.set_title("Aggregate Household Composition")
    ax.legend()
    ax.grid(axis="y")

    fig.tight_layout()
    return fig