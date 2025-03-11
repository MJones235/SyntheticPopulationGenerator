from typing import Dict
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_household_size(synthetic: str, census: Dict):
    synthetic = json.loads(synthetic)
    synthetic = {int(k): v for k, v in synthetic.items()}
    all_sizes = sorted(set(synthetic.keys()).union(set(census.keys())))
    synthetic = {size: synthetic.get(size, 0.0) for size in all_sizes}
    census = {size: census.get(size, 0.0) for size in all_sizes}

    fig, ax = plt.subplots(figsize=(10, 5))

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