from typing import Any, Dict, List
from src.repositories.experiments_repository import ExperimentsRepository

class ExperimentService:
    experiments_repository: ExperimentsRepository

    def __init__(self):
        self.experiments_repository = ExperimentsRepository()
        
    def get(self) -> List[Dict[str, Any]]:
        return self.experiments_repository.get_all_experiments()
    
    def save_experiment(self, experiment: Dict[str, Any]):
        """Inserts experiment into the database."""
        return self.experiments_repository.insert(experiment)
