import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data/evaluation/household_size/resampled_all_locations.csv")

df = df.rename(columns={
    "Upper tier local authorities": "Location",
    "Household size (9 categories)": "Category",
    "Percentage": "Value"
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Category", y="Value")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Household Size")  
plt.ylabel("Percentage")
plt.title("Variation in Ground Truth Age Distribution Across Locations")
#plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.5))
plt.grid(which='major', axis='y', linestyle='--', alpha=0.7)
plt.grid(which='minor', axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()
