from scipy.spatial.distance import jensenshannon
import numpy as np
import pandas as pd

from src.classifiers.household_size.base import HouseholdSizeClassifier
from src.classifiers.household_size.uk_census import UKHouseholdSizeClassifier
from src.classifiers.household_type.base import HouseholdCompositionClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier
from src.services.file_service import FileService
from src.analysis.distributions import compute_occupation_distribution
from src.utils.age_bands import assign_age_band, get_age_band_labels


def compute_metrics(synthetic_counts, census_counts):
    P = np.array(synthetic_counts, dtype=np.float64)
    Q = np.array(census_counts, dtype=np.float64)

    P /= P.sum()
    Q /= Q.sum()

    jsd = jensenshannon(P, Q, base=2) ** 2
    tvd = 0.5 * np.abs(P - Q).sum()
    rmse = np.sqrt(np.mean((P - Q) ** 2))

    return {'JSD': jsd, 'TVD': tvd, 'RMSE': rmse}

def get_synthetic_age_pyramid(df: pd.DataFrame):
    synthetic_df = df.copy()
    synthetic_df["age_group"] = assign_age_band(synthetic_df["age"])
    synthetic_df["gender"] = synthetic_df["gender"].str.capitalize()
    synthetic_df["count"] = 1

    # Aggregate synthetic data by age_group and gender
    syn_grouped = (
        synthetic_df.groupby(["age_group", "gender"], observed=False)["count"].sum().unstack().fillna(0)
    )
    syn_pct = syn_grouped.divide(syn_grouped.sum().sum()).multiply(100)
    _, age_labels = get_age_band_labels()
    return syn_pct.reindex(age_labels).fillna(0)

def get_census_age_pyramid(df: pd.DataFrame):
    _, age_labels = get_age_band_labels()

    census_df = df.copy()
    census_df["age_group"] = census_df["age_group"].str.strip()

    # Extract lower bound of age (e.g., from "25–29" get 25, from "85+" get 85)
    census_df["numeric_age"] = (
        census_df["age_group"]
        .str.extract(r"^(\d+)", expand=False)
        .astype(float)
    )

    # Assign new broader age bands
    census_df["broad_age_band"] = assign_age_band(census_df["numeric_age"])

    # Group by new age band and sum percentages
    grouped = census_df.groupby("broad_age_band", observed=False)[["Male", "Female"]].sum()

    # Reindex to standard order
    grouped = grouped.reindex(age_labels).fillna(0)

    return grouped

def compute_similarity_metrics(df: pd.DataFrame, location: str, include_occupation: bool, hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier(), hh_size_classifier: HouseholdSizeClassifier = UKHouseholdSizeClassifier()):
    hh_size_synth = list(hh_size_classifier.compute_observed_distribution(df).values())
    hh_size_census = list(FileService().load_household_size(location).values())
    hh_size_result = compute_metrics(hh_size_synth, hh_size_census)
    
    age_synth = get_synthetic_age_pyramid(df).stack().to_list()
    age_census = get_census_age_pyramid(FileService().load_age_pyramid(location)).stack().to_list()
    age_result = compute_metrics(age_synth, age_census)

    if include_occupation:
        occupation_synth_dict = compute_occupation_distribution(df)
        occupation_census_dict = FileService().load_occupation_distribution(location)
        all_categories = sorted(set(occupation_synth_dict.keys()).union(set(occupation_census_dict.keys())))
        occupation_synth = list({size: occupation_synth_dict.get(size, 0.0) for size in all_categories}.values())
        occupation_census = list({size: occupation_census_dict.get(size, 0.0) for size in all_categories}.values())
        occupation_result = compute_metrics(occupation_synth, occupation_census)

    hh_type_synth = hh_type_classifier.compute_observed_distribution(df)
    hh_type_census = FileService().load_household_composition(location)

    combined = pd.DataFrame(
        {"Synthetic": hh_type_synth, "Census": hh_type_census}
    ).fillna(0)
    label_order = hh_type_classifier.get_label_order()
    combined = combined.loc[[label for label in label_order if label in combined.index]]
    hh_type_result = compute_metrics(combined['Synthetic'].tolist(), combined['Census'].tolist())

    results = [
        {'Variable': 'Household Size', **hh_size_result},
        {'Variable': 'Age/Gender Pyramid', **age_result},
        {'Variable': 'Household Composition', **hh_type_result}
    ]

    if include_occupation:
        results.append({'Variable': 'Occupation', **occupation_result})

    return pd.DataFrame(results)


def compute_aggregate_metrics(populations: list[pd.DataFrame], location: str, include_occupation: bool,  hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier(), hh_size_classifier: HouseholdSizeClassifier = UKHouseholdSizeClassifier()) -> pd.DataFrame:
    all_metrics = []

    for df in populations:
        try:
            metrics = compute_similarity_metrics(df, location, include_occupation, hh_type_classifier, hh_size_classifier)
            all_metrics.append(metrics.set_index("Variable"))
        except Exception:
            continue

    if not all_metrics:
        return pd.DataFrame()

    # Each metric is a DataFrame with index=Variable, columns=["JSD", "TVD"]
    # Stack them into two lists: one per metric
    jsd_matrix = pd.DataFrame([m["JSD"] for m in all_metrics])
    tvd_matrix = pd.DataFrame([m["TVD"] for m in all_metrics])
    rmse_matrix = pd.DataFrame([m["RMSE"] for m in all_metrics])

    result = pd.DataFrame({
        "Variable": jsd_matrix.columns,
        "Mean JSD": jsd_matrix.mean(),
        "95% CI JSD": 1.96 * jsd_matrix.std() / np.sqrt(len(jsd_matrix)),
        "Mean TVD": tvd_matrix.mean(),
        "95% CI TVD": 1.96 * tvd_matrix.std() / np.sqrt(len(tvd_matrix)),
        "Mean RMSE": rmse_matrix.mean(),
        "95% CI RMSE": 1.96 * rmse_matrix.std() / np.sqrt(len(rmse_matrix)),
    }).reset_index(drop=True)

    return result


def compute_convergence_curve(df, location, step=200, max_points=10000, include_occupation: bool = True, hh_type_classifier: HouseholdCompositionClassifier = UKHouseholdCompositionClassifier(), hh_size_classifier: HouseholdSizeClassifier = UKHouseholdSizeClassifier()):
    results = []

    for i in range(step, min(len(df), max_points), step):
        partial_df = df.iloc[:i]
        try:
            metrics_df = compute_similarity_metrics(partial_df, location, include_occupation, hh_type_classifier, hh_size_classifier)
            metrics_df["n_individuals"] = i
            results.append(metrics_df)
        except Exception as e:
            #print(f"Error at {i} individuals: {e}")
            pass
    if not results:
        return pd.DataFrame()
    
    return pd.concat(results).reset_index(drop=True)
