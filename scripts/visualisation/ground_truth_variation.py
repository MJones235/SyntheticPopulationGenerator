import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/evaluation/age_distribution_2/resampled_data.csv")

df = df.rename(columns={
    "Upper tier local authorities": "Location",
    "Age Bin": "Category",  # or "Household Size" if applicable
    "Percentage": "Value"
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Category", y="Value")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Age Band")  # or "Household Size" if applicable
plt.ylabel("Percentage")
plt.title("Variation in Ground Truth Age Distribution Across Locations")

plt.yticks(np.arange(0, df["Value"].max() + 0.5, 0.5))
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
