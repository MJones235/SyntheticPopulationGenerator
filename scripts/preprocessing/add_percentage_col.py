import pandas as pd

# Load the raw data
df = pd.read_csv("data/evaluation/household_size/all_locations.csv")

# Compute total households per local authority
totals = df.groupby(
    ["Upper tier local authorities Code", "Upper tier local authorities"]
)["Observation"].transform("sum")

# Add a percentage column
df["Percentage"] = (df["Observation"] / totals) * 100
df["Percentage"] = df["Percentage"].round(2)

# Save the result to a new file
df.to_csv("data/evaluation/household_size/resampled_all_locations.csv", index=False)
