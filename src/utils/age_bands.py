import pandas as pd


def get_age_band_labels() -> tuple[list[int], list[str]]:
    """Returns bin edges and labels for broad demographic age groups."""
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, float("inf")]
    labels = [
        "0–9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80+",
    ]
    return bins, labels


def assign_age_band(age_series: pd.Series) -> pd.Series:
    """Assigns age bands to a series of ages using predefined bins."""
    bins, labels = get_age_band_labels()
    return pd.cut(age_series, bins=bins, labels=labels, right=False)
