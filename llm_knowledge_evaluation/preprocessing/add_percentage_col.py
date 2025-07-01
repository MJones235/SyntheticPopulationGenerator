import pandas as pd

# Load the raw data
df = pd.read_csv("data/evaluation/household_size/raw_data.csv")

# Compute total households per local authority
totals = df.groupby(
    ["Upper tier local authorities Code", "Upper tier local authorities"]
)["Observation"].transform("sum")

# Add a percentage column
df["Percentage"] = (df["Observation"] / totals) * 100
df["Percentage"] = df["Percentage"].round(2)

# Save the result to a new file
df.to_csv("data/evaluation/household_size/sampled_data.csv", index=False)
