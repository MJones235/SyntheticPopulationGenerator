import pandas as pd


def estimate_household_size_from_type(type_code: int) -> float:
    mapping = {
        1: 1.0,   # One-person household
        2: 3.75,   # Married/civil partnership couple household
        3: 3.75,   # Cohabiting couple household 
        4: 2.75,  # Lone parent household (1 adult + ~1.75 kids)
        5: 3,   # Multi-person household (flatshares, multigen)
        -8: 3   # Missing / fallback
    }
    return mapping.get(type_code, 3.0)


def sample_microdata(microdata_df: pd.DataFrame, n: int) -> pd.DataFrame:
    microdata_df["estimated_size"] = microdata_df["hh_families_type_6a"].apply(estimate_household_size_from_type)
    microdata_df["sampling_weight"] = 1 / microdata_df["estimated_size"]
    return microdata_df.sample(
        n=n,
        weights=microdata_df["sampling_weight"],
        replace=False
    )