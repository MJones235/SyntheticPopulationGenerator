from abc import ABC, abstractmethod
from typing import Dict
import pandas as pd

class HouseholdCompositionClassifier(ABC):
    @abstractmethod
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame) -> Dict[str, float]:
        """Computes the observed distribution from synthetic data."""
        pass