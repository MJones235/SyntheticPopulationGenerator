from .base import HouseholdSizeClassifier
from typing import Dict
import pandas as pd


class UNHouseholdSizeClassifier(HouseholdSizeClassifier):
    def get_name(self):
        return 'un_global'

    def compute_observed_distribution(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        household_sizes = synthetic_df.groupby("household_id").size()

        def bucket(size):
            if size == 1:
                return "1"
            elif size <= 3:
                return "2-3"
            elif size <= 5:
                return "4-5"
            else:
                return "6+"

        size_buckets = household_sizes.apply(bucket)
        bucket_counts = size_buckets.value_counts().to_dict()
        total = sum(bucket_counts.values())
        buckets = ["1", "2-3", "4-5", "6+"]
        return {
            b: round((bucket_counts.get(b, 0) / total) * 100, 2) if total > 0 else 0.00
            for b in buckets
        }
