from collections import Counter
from typing import Any, Dict, List


def compute_household_size_distribution_from_individuals(individuals: List[Dict[str, Any]]) -> Dict[int, float]:
    household_sizes = Counter([person["household_id"] for person in individuals])
    household_size_distribution = Counter(household_sizes.values())
    total_households = sum(household_size_distribution.values())
    household_size_percentages = {size: (count / total_households) * 100 for size, count in household_size_distribution.items()} if total_households > 0 else {}
    return household_size_percentages

def compute_household_size_distribution_from_households(households: List[List[Dict[str, Any]]]) -> Dict[int, float]:
    household_sizes = [len(h) for h in households if h]
    size_counts = Counter(household_sizes)
    total_households = sum(size_counts.values())
    size_distribution = {
        size: round(size_counts.get(size, 0) / total_households * 100, 2) if total_households > 0 else 0.00
        for size in range(0, 9)
    }
    return size_distribution
