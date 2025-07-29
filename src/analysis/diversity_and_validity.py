import pandas as pd
from collections import Counter

def compute_individual_diversity(df: pd.DataFrame, columns: list[str]) -> float:
    if df.empty:
        return 0.0

    unique_rows = df[columns].drop_duplicates()
    diversity_percent = 100 * len(unique_rows) / len(df)
    return diversity_percent

def compute_household_structure_diversity(df: pd.DataFrame) -> float:
    if "household_id" not in df.columns or "relationship" not in df.columns:
        return 0.0

    household_roles = (
        df.groupby("household_id")["relationship"]
        .apply(lambda roles: tuple(sorted(Counter(roles).items())))
    )

    n_total = len(household_roles)
    n_unique = household_roles.nunique()

    return 100 * n_unique / n_total if n_total > 0 else 0.0

def compute_generation_validity(df: pd.DataFrame, expected_households: int) -> float:
    if expected_households == 0:
        return 0.0

    actual = df["household_id"].nunique()
    return 100 * actual / expected_households
