from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd

class BaseLoader(ABC):
    def __init__(self, location_dir: Path):
        self.location_dir = location_dir

    @abstractmethod
    def load_file(self, filename: str) -> pd.DataFrame:
        """Load a raw data file and return a DataFrame."""
        pass
