from src.repositories.base_repository import BaseRepository
from typing import Dict, Any, List, Tuple

class ExperimentsRepository(BaseRepository):
    """Handles database operations for the experiments table."""

    def table_name(self) -> str:
        return "experiments"

    def insert_experiment(self, experiment: Dict[str, Any]):
        """Inserts experiment experiment into the database."""
        self.insert(experiment)
    
    def get_all_experiments(self) -> List[Tuple]:
        """Fetches all experiment IDs and timestamps, sorted by newest first."""
        return self.fetch_all("1=1 ORDER BY timestamp DESC", ())
    
    def get_by_id(self, experiment_id: str) -> Tuple:
        return self.fetch_one("experiment_id = ?", (experiment_id,))
