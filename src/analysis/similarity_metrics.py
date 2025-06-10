from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd

from src.analysis.classifiers import classify_household_structure, household_type_labels
from src.services.file_service import FileService
from src.analysis.distributions import compute_household_size_distribution, compute_occupation_distribution
from src.utils.age_bands import assign_age_band, get_age_band_labels


def compute_metrics(synthetic_counts, census_counts):
    P = np.array(synthetic_counts, dtype=np.float64)
    Q = np.array(census_counts, dtype=np.float64)

    P /= P.sum()
    Q /= Q.sum()

    jsd = jensenshannon(P, Q, base=2) ** 2
    tvd = 0.5 * np.abs(P - Q).sum()

    return {'JSD': jsd, 'TVD': tvd}

def get_synthetic_age_pyramid(df: pd.DataFrame):
    synthetic_df = df.copy()
    synthetic_df["age_group"] = assign_age_band(synthetic_df["age"])
    synthetic_df["gender"] = synthetic_df["gender"].str.capitalize()
    synthetic_df["count"] = 1

    # Aggregate synthetic data by age_group and gender
    syn_grouped = (
        synthetic_df.groupby(["age_group", "gender"])["count"].sum().unstack().fillna(0)
    )
    syn_pct = syn_grouped.divide(syn_grouped.sum().sum()).multiply(100)
    _, age_labels = get_age_band_labels()
    return syn_pct.reindex(age_labels).fillna(0)

def get_census_age_pyramid(df: pd.DataFrame):
    _, age_labels = get_age_band_labels()
    census_df = df.copy()
    census_df = census_df.reset_index().rename(columns={"age_group": "raw_group"})
    census_df["numeric_age"] = (
        census_df["raw_group"].str.extract(r"(\d+)", expand=False).astype(float)
    )
    census_df["age_group"] = assign_age_band(census_df["numeric_age"])

    census_grouped = census_df.groupby("age_group")[["Male", "Female"]].sum()
    census_pct = census_grouped.divide(census_grouped.sum().sum()).multiply(100)

    return census_pct.reindex(age_labels).fillna(0)

def get_synthetic_household_composition(df: pd.DataFrame):
    label_map, _ = household_type_labels()
    household_labels = df.groupby("household_id").apply(classify_household_structure)
    synthetic_counts = household_labels.value_counts(normalize=True) * 100
    synthetic_counts.index = synthetic_counts.index.map(lambda x: label_map.get(x, x))
    return synthetic_counts


def get_census_household_composition(df: pd.DataFrame):
    census_df = df.copy()
    label_map, _ = household_type_labels()
    census_df["Short Label"] = census_df["Household Composition"].map(
    lambda x: label_map.get(x, x)
    )
    census_df = census_df.groupby("Short Label")["Value"].sum().reset_index()
    return census_df.groupby("Short Label")["Value"].sum()


def compute_similarity_metrics(df: pd.DataFrame, location: str):
    hh_size_synth = list(compute_household_size_distribution(df).values())
    hh_size_census = list(FileService().load_household_size(location).values())
    hh_size_result = compute_metrics(hh_size_synth, hh_size_census)

    age_synth = get_synthetic_age_pyramid(df).stack().to_list()
    age_census = get_census_age_pyramid(FileService().load_age_pyramid(location)).stack().to_list()
    age_result = compute_metrics(age_synth, age_census)

    occupation_synth_dict = compute_occupation_distribution(df)
    occupation_census_dict = FileService().load_occupation_distribution(location)
    all_categories = sorted(set(occupation_synth_dict.keys()).union(set(occupation_census_dict.keys())))
    occupation_synth = list({size: occupation_synth_dict.get(size, 0.0) for size in all_categories}.values())
    occupation_census = list({size: occupation_census_dict.get(size, 0.0) for size in all_categories}.values())

    occupation_result = compute_metrics(occupation_synth, occupation_census)

    hh_type_synth = get_synthetic_household_composition(df)
    hh_type_census = get_census_household_composition(FileService().load_household_composition(location))
    combined = pd.DataFrame(
        {"Synthetic": hh_type_synth, "Census": hh_type_census}
    ).fillna(0)
    _, label_order = household_type_labels()
    combined = combined.loc[[label for label in label_order if label in combined.index]]
    hh_type_result = compute_metrics(combined['Synthetic'].tolist(), combined['Census'].tolist())

    return pd.DataFrame([
        {'Variable': 'Household Size', **hh_size_result},
        {'Variable': 'Age/Gender Pyramid', **age_result},
        {'Variable': 'Occupation', **occupation_result},
        {'Variable': 'Household Composition', **hh_type_result}
    ])


def compute_aggregate_metrics(populations: list[pd.DataFrame], location: str) -> pd.DataFrame:
    all_metrics = []

    for df in populations:
        try:
            metrics = compute_similarity_metrics(df, location)
            all_metrics.append(metrics.set_index("Variable"))
        except Exception:
            continue

    if not all_metrics:
        return pd.DataFrame()

    # Each metric is a DataFrame with index=Variable, columns=["JSD", "TVD"]
    # Stack them into two lists: one per metric
    jsd_matrix = pd.DataFrame([m["JSD"] for m in all_metrics])
    tvd_matrix = pd.DataFrame([m["TVD"] for m in all_metrics])

    result = pd.DataFrame({
        "Variable": jsd_matrix.columns,
        "Mean JSD": jsd_matrix.mean(),
        "95% CI JSD": 1.96 * jsd_matrix.std() / np.sqrt(len(jsd_matrix)),
        "Mean TVD": tvd_matrix.mean(),
        "95% CI TVD": 1.96 * tvd_matrix.std() / np.sqrt(len(tvd_matrix)),
    }).reset_index(drop=True)

    return result
