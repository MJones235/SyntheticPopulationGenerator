class UNCountryRegistry:
    def __init__(self):
        self.country_info = {}

    def register(self, raw_name: str, canonical_name: str, population: float):
        self.country_info[raw_name.strip()] = {
            "canonical": canonical_name.strip(),
            "population": population
        }

    def get_canonical(self, raw_name: str) -> str:
        return self.country_info.get(raw_name.strip(), {}).get("canonical", raw_name.strip())

    def has_population_over(self, raw_name: str, threshold=100_000) -> bool:
        return self.country_info.get(raw_name.strip(), {}).get("population", 0) >= threshold

    def all_countries(self):
        return self.country_info.keys()
