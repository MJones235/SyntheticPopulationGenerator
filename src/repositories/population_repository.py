from src.repositories.base_repository import BaseRepository
from typing import List, Dict, Any
import uuid

class PopulationRepository(BaseRepository):
    """Handles database operations for the populations table."""
    
    def table_name(self) -> str:
        return "populations"
    
    def insert_population(self, population_id: str, households: List[List[Dict[str, Any]]]):
        """Inserts multiple households into the populations table."""
        for household in households:
            household_id = str(uuid.uuid4())
            for person in household:
                self.insert({
                    "id": str(uuid.uuid4()),
                    "population_id": population_id,
                    "household_id": household_id,
                    "name": person.get("name", ""),
                    "age": person.get("age", -1),
                    "gender": person.get("gender", ""),
                    "occupation_category": person.get("occupation_category", -1),
                    "occupation": person.get("occupation", ""),
                    "relationship": person.get("relationship_to_head", "")
                })

    def get_population_by_id(self, population_id: str) -> List[Dict[str, Any]]:
        """Fetches all individuals belonging to a specific population."""
        return self.fetch_all("population_id = ?", (population_id,))
