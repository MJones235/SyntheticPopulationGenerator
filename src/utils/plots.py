from typing import Dict
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd

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
    census_groups = census_df['age_group'].unique().tolist()

    def assign_age_group_column(df: pd.DataFrame) -> pd.Series:
        bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
            55, 60, 65, 70, 75, 80, 85, 90, float('inf')]
        labels = [
            "0–4", "5–9", "10–14", "15–19", "20–24", "25–29", "30–34",
            "35–39", "40–44", "45–49", "50–54", "55–59", "60–64",
            "65–69", "70–74", "75–79", "80–84", "85–89", "90+"
        ]

        return pd.cut(df["age"], bins=bins, labels=labels, right=False)
        
    synthetic_df["age_group"] = assign_age_group_column(synthetic_df)
    synthetic_df['gender'] = synthetic_df['gender'].str.capitalize()
    synthetic_df['count'] = 1
    
    # Synthetic population
    syn_grouped = synthetic_df.groupby(['age_group', 'gender'])['count'].sum().unstack().fillna(0)
    syn_pct = syn_grouped.divide(syn_grouped.sum().sum()).multiply(100)
    syn_pct = syn_pct.reindex(census_groups) 
    census_pct = census_df.set_index("age_group").reindex(census_groups).fillna(0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot synthetic (bold, foreground)
    ax.barh(syn_pct.index, -syn_pct.get("Male", 0), color="blue", label="Synthetic Male")
    ax.barh(syn_pct.index, syn_pct.get("Female", 0), color="red", label="Synthetic Female")

    # Plot census (light, behind, with edge)
    ax.barh(
        census_pct.index,
        -census_pct.get("Male", 0),
        color="lightblue",
        alpha=0.4,
        edgecolor="black",
        label="Census Male"
    )
    ax.barh(
        census_pct.index,
        census_pct.get("Female", 0),
        color="pink",
        alpha=0.4,
        edgecolor="black",
        label="Census Female"
    )

    ax.axvline(0, color="black")
    ax.set_xlabel("Percentage of Population")
    ax.set_ylabel("Age Group")
    ax.legend(loc="lower right")
    ax.set_title("Age Pyramid: Synthetic vs Census")

    return fig
