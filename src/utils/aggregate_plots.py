import json
import numpy as np
import matplotlib.pyplot as plt

def plot_household_size_aggregate(synthetic_runs: list[str], census: dict):
    parsed_runs = []
    for run in synthetic_runs:
        try:
            d = json.loads(run)
            parsed_runs.append({int(k): v for k, v in d.items()})
        except (json.JSONDecodeError, ValueError):
            continue  # skip malformed
    
    if not parsed_runs:
        raise ValueError("No valid household size data to aggregate.")

    # Collect all keys
    all_keys = sorted(set().union(*[set(d.keys()) for d in parsed_runs], set(census.keys())))
    all_keys = list(map(int, all_keys))

    # Normalize and align
    data = []
    for run in parsed_runs:
        normed = {k: run.get(k, 0.0) for k in all_keys}
        print(run)
        print(normed)
        data.append([normed[k] for k in all_keys])

    print(data)

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
    ax.bar(indices, mean, width=bar_width, yerr=ci, label="Synthetic (mean ± CI)", color="orange", alpha=0.6)

    ax.set_xticks(indices)
    ax.set_xticklabels(all_keys)
    ax.set_xlabel("Household Size")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Aggregate Household Size Distribution")
    ax.legend()
    return fig
