from collections import Counter
from typing import Any, Dict, List
from src.repositories.population_repository import PopulationRepository
from src.repositories.analysis_repository import AnalysisRepository

class AnalysisService:
    analysis_repository: AnalysisRepository
    population_repository: PopulationRepository

    def __init__(self):
        self.analysis_repository = AnalysisRepository()
        self.population_repository = PopulationRepository()
    
    def get_by_id(self, population_id: str) -> Dict[str, Any]:
        return self.analysis_repository.get_analysis(population_id)
        
    def save_analysis(self, population_id: str):
        """Inserts experiment metadata into the database."""
        return self.analysis_repository.insert_analysis(population_id, self.compute_household_size_distribution(population_id))

    def compute_household_size_distribution(self, population_id: str) -> Dict[int, float]:
        """Computes household size distribution as percentages for a given synthetic population."""
        
        individuals = self.population_repository.get_population_by_id(population_id)
        if not individuals:
            return {}

        household_sizes = Counter([person["household_id"] for person in individuals])
        household_size_distribution = Counter(household_sizes.values())
        total_households = sum(household_size_distribution.values())
        household_size_percentages = {size: (count / total_households) * 100 for size, count in household_size_distribution.items()}
        return household_size_percentages
