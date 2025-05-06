from typing import Dict
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

from src.utils.age_bands import assign_broad_age_band, get_broad_age_band_labels

def plot_household_size(synthetic: str, census: Dict):
    synthetic = json.loads(synthetic)
    synthetic = {int(k): v for k, v in synthetic.items()}
    all_sizes = sorted(set(synthetic.keys()).union(set(census.keys())))
    synthetic = {size: synthetic.get(size, 0.0) for size in all_sizes}
    census = {size: census.get(size, 0.0) for size in all_sizes}

    fig, ax = plt.subplots(figsize=(8, 3))

    bar_width = 0.4 
    indices = np.arange(len(all_sizes)) 

    ax.bar(indices - bar_width / 2, census.values(), bar_width, label="Census Data", color="blue", alpha=0.6)
    ax.bar(indices + bar_width / 2, synthetic.values(), bar_width, label="Synthetic Data", color="orange", alpha=0.6)

    ax.set_xlabel("Household Size")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Comparison of Household Size Distribution")
    ax.set_xticks(indices)
    ax.set_xticklabels(all_sizes) 
    ax.legend()

    return fig

def plot_age_pyramid(synthetic_df: pd.DataFrame, census_df: pd.DataFrame):
    # Assign broad age bands to synthetic population
    synthetic_df = synthetic_df.copy()
    synthetic_df["age_group"] = assign_broad_age_band(synthetic_df["age"])
    synthetic_df["gender"] = synthetic_df["gender"].str.capitalize()
    synthetic_df["count"] = 1

    # Aggregate synthetic data by age_group and gender
    syn_grouped = synthetic_df.groupby(["age_group", "gender"])["count"].sum().unstack().fillna(0)
    syn_pct = syn_grouped.divide(syn_grouped.sum().sum()).multiply(100)

    # Use the same broad bands and align census
    bins, age_labels = get_broad_age_band_labels()
    census_df = census_df.copy()
    census_df = census_df.reset_index().rename(columns={"age_group": "raw_group"})
    census_df["numeric_age"] = census_df["raw_group"].str.extract(r"(\d+)", expand=False).astype(float)
    census_df["age_group"] = assign_broad_age_band(census_df["numeric_age"])

    census_grouped = census_df.groupby("age_group")[["Male", "Female"]].sum()
    census_pct = census_grouped.divide(census_grouped.sum().sum()).multiply(100)

    # Reindex both to ensure matching age group order
    syn_pct = syn_pct.reindex(age_labels).fillna(0)
    census_pct = census_pct.reindex(age_labels).fillna(0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Synthetic (solid)
    ax.barh(syn_pct.index, -syn_pct.get("Male", 0), color="blue", label="Synthetic Male")
    ax.barh(syn_pct.index, syn_pct.get("Female", 0), color="red", label="Synthetic Female")

    # Census (shaded)
    ax.barh(
        census_pct.index, -census_pct.get("Male", 0),
        color="lightblue", alpha=0.4, edgecolor="black", label="Census Male"
    )
    ax.barh(
        census_pct.index, census_pct.get("Female", 0),
        color="pink", alpha=0.4, edgecolor="black", label="Census Female"
    )

    ax.axvline(0, color="black")
    ax.set_xlabel("Percentage of Population")
    ax.set_ylabel("Age Group")
    ax.set_title("Age Pyramid: Synthetic vs Census")
    ax.legend(loc="lower right")
    return fig
