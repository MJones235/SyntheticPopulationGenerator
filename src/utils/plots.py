from typing import Dict
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from src.analysis.similarity_metrics import get_census_age_pyramid, get_census_household_composition, get_synthetic_age_pyramid, get_synthetic_household_composition
from src.analysis.classifiers import classify_household_structure, household_type_labels
from src.utils.age_bands import assign_age_band, get_age_band_labels


def plot_household_size(synthetic: str, census: Dict):
    synthetic = json.loads(synthetic)
    synthetic = {int(k): v for k, v in synthetic.items()}
    all_sizes = sorted(set(synthetic.keys()).union(set(census.keys())))
    synthetic = {size: synthetic.get(size, 0.0) for size in all_sizes}
    census = {size: census.get(size, 0.0) for size in all_sizes}

    fig, ax = plt.subplots(figsize=(8, 3))

    bar_width = 0.4
    indices = np.arange(len(all_sizes))

    ax.bar(
        indices - bar_width / 2,
        census.values(),
        bar_width,
        label="Census Data",
        color="blue",
        alpha=0.6,
    )
    ax.bar(
        indices + bar_width / 2,
        synthetic.values(),
        bar_width,
        label="Synthetic Data",
        color="orange",
        alpha=0.6,
    )

    ax.set_xlabel("Household Size")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison of Household Size Distribution")
    ax.set_xticks(indices)
    ax.set_xticklabels(all_sizes)
    ax.legend()

    return fig


def plot_age_pyramid(synthetic_df: pd.DataFrame, census_df: pd.DataFrame):

    syn_pct = get_synthetic_age_pyramid(synthetic_df)
    census_pct = get_census_age_pyramid(census_df)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Synthetic (solid)
    ax.barh(
        syn_pct.index, -syn_pct.get("Male", 0), color="blue", label="Synthetic Male"
    )
    ax.barh(
        syn_pct.index, syn_pct.get("Female", 0), color="red", label="Synthetic Female"
    )

    # Census (shaded)
    ax.barh(
        census_pct.index,
        -census_pct.get("Male", 0),
        color="lightblue",
        alpha=0.4,
        edgecolor="black",
        label="Census Male",
    )
    ax.barh(
        census_pct.index,
        census_pct.get("Female", 0),
        color="pink",
        alpha=0.4,
        edgecolor="black",
        label="Census Female",
    )

    ax.axvline(0, color="black")
    ax.set_xlabel("Percentage of Population")
    ax.set_ylabel("Age Group")
    ax.set_title("Age Pyramid: Synthetic vs Census")
    ax.legend(loc="lower right")
    return fig


def plot_occupations(synthetic: dict, census: dict):
    all_categories = sorted(set(synthetic.keys()).union(set(census.keys())))
    synthetic = {size: synthetic.get(size, 0.0) for size in all_categories}
    census = {size: census.get(size, 0.0) for size in all_categories}

    fig, ax = plt.subplots(figsize=(8, 3))

    bar_width = 0.4
    indices = np.arange(len(all_categories))

    ax.bar(
        indices - bar_width / 2,
        census.values(),
        bar_width,
        label="Census Data",
        color="blue",
        alpha=0.6,
    )
    ax.bar(
        indices + bar_width / 2,
        synthetic.values(),
        bar_width,
        label="Synthetic Data",
        color="orange",
        alpha=0.6,
    )

    ax.set_xlabel("Standard Occupation Category")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison of Occupation Distribution")
    ax.set_xticks(indices)
    ax.set_xticklabels(all_categories)
    ax.legend()

    return fig


def plot_age_diff(synthetic_df: pd.DataFrame):
    head_child_diffs = []
    head_partner_diffs = []

    for _, group in synthetic_df.groupby("household_id"):
        head = group[group["relationship"] == "Head"]
        children = group[group["relationship"] == "Child"]
        partners = group[group["relationship"].isin(["Partner", "Spouse"])]

        if not head.empty:
            head_age = head.iloc[0]["age"]

            # Head–Child
            for _, child in children.iterrows():
                diff = head_age - child["age"]
                head_child_diffs.append(diff)

            # Head–Partner
            for _, partner in partners.iterrows():
                diff = head_age - partner["age"]
                head_partner_diffs.append(diff)

    # Create the figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    axes[0].hist(head_child_diffs, bins=15, color="lightblue", edgecolor="black")
    axes[0].set_title("Head–Child Age Differences")
    axes[0].set_xlabel("Years")
    axes[0].set_ylabel("Count")
    axes[0].grid(True)

    axes[1].hist(head_partner_diffs, bins=15, color="salmon", edgecolor="black")
    axes[1].set_title("Head-Partner Age Differences")
    axes[1].set_xlabel("Years")
    axes[1].grid(True)

    fig.suptitle("Intra-Household Age Differences")
    fig.tight_layout()

    return fig


def plot_household_structure_bar(
    df: pd.DataFrame, census_df: pd.DataFrame
) -> plt.Figure:
    _, label_order = household_type_labels()
    synthetic_counts = get_synthetic_household_composition(df)
    census_counts = get_census_household_composition(census_df)

    combined = pd.DataFrame(
        {"Synthetic": synthetic_counts, "Census": census_counts}
    ).fillna(0)

    combined = combined.loc[[label for label in label_order if label in combined.index]]


    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.4
    x = range(len(combined))
    ax.bar(
        [i - bar_width / 2 for i in x],
        combined["Census"],
        width=bar_width,
        label="Census",
        color="royalblue",
    )
    ax.bar(
        [i + bar_width / 2 for i in x],
        combined["Synthetic"],
        width=bar_width,
        label="Synthetic",
        color="orange",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(combined.index, rotation=45, ha="right")
    ax.set_ylabel("Percentage of Households")
    ax.set_title("Comparison of Household Composition: Census vs Synthetic")
    ax.legend()
    ax.grid(axis="y")

    return fig


def plot_occupation_titles(df: pd.DataFrame) -> plt.Figure:
    top_occupations = df["occupation"].value_counts().head(25)

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        top_occupations.index[::-1],
        top_occupations.values[::-1],
        color="steelblue",
        edgecolor="black",
    )
    ax.set_title("Top 25 Occupations")
    ax.set_xlabel("Number of Individuals")
    ax.set_ylabel("Occupation Title")
    ax.grid(axis="x")

    return fig
