from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class HouseholdSizeClassifier(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """Computes the observed distribution from synthetic data."""
        pass

    def compute_average_household_size(self, synthetic_df: pd.DataFrame) -> float:
        """Computes the average household size from synthetic data."""
        household_sizes = synthetic_df.groupby("household_id").size()
        return household_sizes.mean() if not household_sizes.empty else 0.0