from src.repositories.base_repository import BaseRepository

class EstimationRepository(BaseRepository):
    def table_name(self) -> str:
        return "estimations"

    def insert_estimation(self, estimation: dict):
        self.insert(estimation)
