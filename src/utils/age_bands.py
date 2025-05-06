import pandas as pd

def get_broad_age_band_labels() -> tuple[list[int], list[str]]:
    """Returns bin edges and labels for broad demographic age groups."""
    bins = [0, 15, 25, 45, 65, 80, float("inf")]
    labels = ["0–14", "15–24", "25–44", "45–64", "65–79", "80+"]
    return bins, labels


def assign_broad_age_band(age_series: pd.Series) -> pd.Series:
    """Assigns broad age bands to a series of ages using predefined bins."""
    bins, labels = get_broad_age_band_labels()
    return pd.cut(age_series, bins=bins, labels=labels, right=False)
