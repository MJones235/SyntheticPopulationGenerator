import pandas as pd


# Load the raw data
df = pd.read_csv("data/evaluation/age_distribution_2/raw_data.csv")

# Convert the 'Age (86 categories) Code' column to string and extract numeric age
df["Numeric Age"] = df["Age (86 categories) Code"].astype(str).str.extract(r"(\d+)").astype(float)

# Define age binning function
def assign_age_bin(age):
    if age < 10:
        return "0-9"
    elif age < 20:
        return "10-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80+"

# Apply age binning
df["Age Bin"] = df["Numeric Age"].apply(assign_age_bin)

# Group by local authority and age bin, summing the observations
resampled_df = df.groupby(
    ["Upper tier local authorities Code", "Upper tier local authorities", "Age Bin"]
)["Observation"].sum().reset_index()

totals = resampled_df.groupby(
    ["Upper tier local authorities Code", "Upper tier local authorities"]
)["Observation"].transform("sum")

resampled_df["Percentage"] = (resampled_df["Observation"] / totals) * 100
resampled_df["Percentage"] = resampled_df["Percentage"].round(2)

# Save the resampled data to a new CSV file
resampled_df.to_csv("data/evaluation/age_distribution_2/resampled_data.csv", index=False)
