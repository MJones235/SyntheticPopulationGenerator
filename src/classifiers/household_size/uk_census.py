from .base import HouseholdSizeClassifier
from typing import Dict
import pandas as pd


class UKHouseholdSizeClassifier(HouseholdSizeClassifier):
    def get_name(self):
        return 'uk_census'

    def compute_observed_distribution(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        household_sizes = synthetic_df.groupby("household_id").size()
        household_sizes = household_sizes.apply(lambda x: x if x <= 8 else 8)
        size_counts = household_sizes.value_counts().to_dict()
        total = sum(size_counts.values())
        return {
            size: round((size_counts.get(size, 0) / total) * 100, 2) if total > 0 else 0.00
            for size in range(1, 9)
        } 
