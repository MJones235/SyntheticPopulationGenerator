import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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