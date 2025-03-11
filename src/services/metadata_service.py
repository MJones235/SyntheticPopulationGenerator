from typing import Any, Dict, List
from src.repositories.metadata_repository import MetadataRepository

class MetadataService:
    metadata_repository: MetadataRepository

    def __init__(self):
        self.metadata_repository = MetadataRepository()
    
    def get_by_id(self, population_id: str) -> Dict[str, Any]:
        return self.metadata_repository.get_metadata_by_population_id(population_id)
    
    def get(self) -> List[Dict[str, Any]]:
        return self.metadata_repository.get_all_populations()
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Inserts experiment metadata into the database."""
        return self.metadata_repository.insert(metadata)
