import pandas as pd

from src.utils.age_bands import assign_broad_age_band

def compute_household_size_distribution(df: pd.DataFrame) -> dict:
    household_sizes = df.groupby("household_id").size()
    size_counts = household_sizes.value_counts().to_dict()
    total = sum(size_counts.values())

    return {
        size: round((size_counts.get(size, 0) / total) * 100, 2) if total > 0 else 0.00
        for size in range(1, 9)
    }

def compute_gender_distribution(df: pd.DataFrame) -> dict:
    dist = df["gender"].str.capitalize().value_counts(normalize=True) * 100
    return dist.round(1).to_dict()

def compute_broad_age_distribution(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["age_band"] = assign_broad_age_band(df["age"])
    dist = df["age_band"].value_counts(normalize=True) * 100
    return dist.round(1).to_dict()

def compute_target_broad_age_distribution(census_df: pd.DataFrame) -> dict:
    df = census_df.copy().reset_index().rename(columns={"age_group": "raw_band"})
    df["numeric_age"] = df["raw_band"].str.extract(r"(\d+)", expand=False).astype(float)
    df["age_band"] = assign_broad_age_band(df["numeric_age"])

    age_totals = df.groupby("age_band")[["Male", "Female"]].sum().sum(axis=1)
    distribution = (age_totals / age_totals.sum() * 100).round(1)

    return distribution.to_dict()
