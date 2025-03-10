from src.repositories.base_repository import BaseRepository
from typing import Dict, Any, List, Tuple

class MetadataRepository(BaseRepository):
    """Handles database operations for the metadata table."""

    def table_name(self) -> str:
        return "metadata"

    def insert_metadata(self, metadata: Dict[str, Any]):
        """Inserts experiment metadata into the database."""
        self.insert(metadata)

    def get_metadata_by_population_id(self, population_id: str) -> Tuple:
        """Fetches metadata for a specific population."""
        return self.fetch_one("population_id = ?", (population_id,))
    
    def get_all_populations(self) -> List[Tuple]:
        """Fetches all population IDs and timestamps, sorted by newest first."""
        return self.fetch_all("1=1 ORDER BY timestamp DESC", ())
