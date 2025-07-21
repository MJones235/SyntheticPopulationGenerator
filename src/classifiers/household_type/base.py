from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd

class HouseholdCompositionClassifier(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def compute_observed_distribution(self, synthetic_df: pd.DataFrame, relationship_col: str) -> Dict[str, float]:
        """Computes the observed distribution from synthetic data."""
        pass

    @abstractmethod
    def get_label_order(self) -> List[str]:
        pass