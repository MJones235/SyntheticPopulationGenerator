from typing import Any, Dict
from src.repositories.experiment_runs_repository import ExperimentRunRepository

class ExperimentRunService:
    experiment_run_repository: ExperimentRunRepository

    def __init__(self):
        self.experiment_run_repository = ExperimentRunRepository()
    
    def get_by_experiment_id(self, experiment_id: str) -> Dict[str, Any]:
        return self.experiment_run_repository.get_runs_by_experiment_id(experiment_id)
        
    def save_run(self, experiment_run: Dict[str, Any]):
        """Inserts experiment run into the database."""
        return self.experiment_run_repository.insert(experiment_run)
