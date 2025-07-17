from .base_loader import BaseLoader
import pandas as pd

class UKCensusLoader(BaseLoader):
    def load_file(self, filename: str) -> pd.DataFrame:
        path = self.location_dir / filename
        return pd.read_csv(path)
