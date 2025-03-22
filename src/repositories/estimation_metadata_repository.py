from src.repositories.base_repository import BaseRepository

class EstimationMetadataRepository(BaseRepository):
    def table_name(self) -> str:
        return "estimation_metadata"

    def insert_metadata(self, metadata: dict):
        self.insert(metadata)
