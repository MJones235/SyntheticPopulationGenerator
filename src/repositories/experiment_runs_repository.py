from src.repositories.base_repository import BaseRepository
from typing import Dict, Any, Tuple

class ExperimentRunRepository(BaseRepository):
    """Handles database operations for the experiment_runs table."""

    def table_name(self) -> str:
        return "experiment_runs"

    def insert_experiment_run(self, experiment_run: Dict[str, Any]):
        """Inserts experiment_run into the database."""
        self.insert(experiment_run)

    def get_runs_by_experiment_id(self, experiment_id: str) -> Tuple:
        """Fetches experiment run for a specific population."""
        return self.fetch_all("experiment_id = ?", (experiment_id,))
    