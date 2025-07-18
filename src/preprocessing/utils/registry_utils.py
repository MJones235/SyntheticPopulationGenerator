import pandas as pd
from src.preprocessing.utils.un_country_registry import UNCountryRegistry

def register_un_countries_from_age_group(df: pd.DataFrame, registry: UNCountryRegistry):
    for country in df["Country"].unique():
        total_pop = df[df["Country"] == country]["Population"].sum() * 1000
        registry.register(raw_name=country, canonical_name=country, population=total_pop)
