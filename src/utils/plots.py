from typing import Dict
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import seaborn as sns

from src.classifiers.household_type.base import HouseholdCompositionClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier
from src.analysis.similarity_metrics import get_census_age_pyramid, get_synthetic_age_pyramid

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


def plot_categories(synthetic: dict, census: dict, x_label: str, title: str):
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

    ax.set_xlabel(x_label)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.set_xticks(indices)
    ax.set_xticklabels(all_categories)
    ax.legend()

    return fig


def get_age_of_partner_of_sex(sex: str, head: pd.DataFrame, partners: pd.DataFrame) -> pd.Series:
    if not head.empty and head.iloc[0]["gender"] == sex:
        return head.iloc[0]["age"]
    elif not partners.empty and partners.iloc[0]["gender"] == sex:
        return partners.iloc[0]["age"]
    else: return None


def plot_age_diff(synthetic_df: pd.DataFrame):
    mother_child_diffs = []
    mother_eldest_child_diffs = []
    father_child_diffs = []
    husband_wife_diffs = []

    for _, group in synthetic_df.groupby("household_id"):
        head = group[group["relationship"] == "Head"]
        children = group[group["relationship"] == "Child"]
        partners = group[group["relationship"].isin(["Partner", "Spouse"])]

        if head.empty:
            continue

        husband = get_age_of_partner_of_sex("Male", head, partners)
        wife = get_age_of_partner_of_sex("Female", head, partners)

        if wife is not None and not children.empty:
            eldest_child_age = children["age"].max()
            mother_eldest_child_diffs.append(wife - eldest_child_age)

        for _, child in children.iterrows():
            if husband is not None:
                father_child_diffs.append(husband - child["age"])
            if wife is not None:
                mother_child_diffs.append(wife - child["age"])
            if husband is not None and wife is not None:
                husband_wife_diffs.append(husband - wife)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    axes[0].hist(mother_child_diffs, bins=15, color="lightgreen", edgecolor="black")
    axes[0].set_title("Mother – Eldest Child Age Difference")
    axes[0].set_xlabel("Years")
    axes[0].set_ylabel("Count")
    axes[0].grid(True)

    axes[1].hist(father_child_diffs, bins=15, color="lightblue", edgecolor="black")
    axes[1].set_title("Father – Eldest Child Age Difference")
    axes[1].set_xlabel("Years")
    axes[1].grid(True)

    axes[2].hist(husband_wife_diffs, bins=15, color="salmon", edgecolor="black")
    axes[2].set_title("Husband – Wife Age Difference")
    axes[2].set_xlabel("Years")
    axes[2].grid(True)

    fig.suptitle("Parental and Spousal Age Differences")
    fig.tight_layout()

    mean_mother_birth_age = round(np.mean(mother_child_diffs), 1) if mother_child_diffs else None
    mean_father_birth_age = round(np.mean(father_child_diffs), 1) if father_child_diffs else None
    median_mother_first_birth_age = round(np.median(mother_eldest_child_diffs), 1) if mother_eldest_child_diffs else None

    return fig, mean_mother_birth_age, mean_father_birth_age, median_mother_first_birth_age



def plot_household_structure_bar(
    df: pd.DataFrame, census_df: pd.DataFrame, hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier()
) -> plt.Figure:
    label_order = hh_type_classifier.get_label_order()
    synthetic_counts = hh_type_classifier.compute_observed_distribution(df)

    combined = pd.DataFrame(
        {"Synthetic": synthetic_counts, "Census": census_df}
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

def plot_jsd_heatmap():
    # Load your metrics file
    df = pd.read_csv("outputs/metrics.csv")  # Adjust path if needed

    # Create a unique label for each combination of Approach, Prompt, and Model
    df["Label"] = (
        df["Approach"].astype(str) +
        df["Prompt"].astype(str) +
        " - " + df["Model"]
    )

    # Define the JSD columns
    jsd_cols = ["JSD_hh_size", "JSD_age", "JSD_occ", "JSD_hh_type"]

    # Group by label and take the mean in case of duplicates
    jsd_data = df.groupby("Label")[jsd_cols].mean()

    # Plot the JSD heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(jsd_data, annot=True, fmt=".3f", cmap="YlGnBu", linewidths=0.5)
    plt.title("Jensen-Shannon Divergence (JSD) Heatmap Across Variables")
    plt.ylabel("Approach–Prompt–Model")
    plt.xlabel("JSD Metric")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()