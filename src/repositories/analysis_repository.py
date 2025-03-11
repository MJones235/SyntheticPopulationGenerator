import json
from typing import Dict
from src.repositories.base_repository import BaseRepository

class AnalysisRepository(BaseRepository):
    """Handles database operations for analysis results."""

    def table_name(self) -> str:
        return "analysis"

    def insert_analysis(self, population_id: str, household_size_distribution: Dict):
        """Stores household size distribution as JSON."""
        distribution_json = json.dumps(household_size_distribution)
        self.insert({
            "population_id": population_id,
            "household_size_distribution": distribution_json
        })

    def get_analysis(self, population_id):
        """Retrieves household size distribution for a population."""
        return self.fetch_one("population_id = ?", (population_id,))
