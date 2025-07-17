from pathlib import Path
import pandas as pd
from config import RAW_DATA_DIR, OUTPUT_DIR
from src.preprocessing.loaders.uk_census_loader import UKCensusLoader
from src.preprocessing.loaders.un_household_loader import UNHouseholdLoader
from src.preprocessing.transformers.un_household_transformer import UNHouseholdTransformer
from src.preprocessing.transformers.uk_census_transformer import UKCensusTransformer
from utils.io import save_processed

def clean_age_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["Age (B) (18 categories)"] = (
        df["Age (B) (18 categories)"]
        .str.replace(r"Aged (\d+) years and under", r"0-\1", regex=True)
        .str.replace(r"Aged (\d+) to (\d+) years", r"\1-\2", regex=True)
        .str.replace(r"Aged (\d+) years and over", r"\1+", regex=True)
        .str.strip()
    )
    return df

def clean_household_composition_labels(df: pd.DataFrame) -> pd.DataFrame:
    label_map = {
        "One-person household: Aged 66 years and over": "One-person aged 66+ years",
        "One-person household: Other": "One-person aged <66 years",
        "Single family household: Lone parent household": "Lone parent",
        "Single family household: Couple family household: No children": "Couple",
        "Single family household: Couple family household: Dependent children": "Couple with dependent children",
        "Single family household: Couple family household: All children non-dependent": "Couple with non-dependent children",
        "Other household types": "Other",
    }

    column = "Household composition (8 categories)"
    df[column] = df[column].map(label_map).fillna(df[column])
    return df

def clean_household_size_labels(df: pd.DataFrame) -> pd.DataFrame:
    column = "Household size (9 categories)"

    df[column] = (
        df[column]
        .str.replace(r"^(\d+) people in household$", r"\1", regex=True)
        .str.replace(r"^(\d+) person in household$", r"\1", regex=True)
        .str.replace(r"^(\d+) or more people in household$", r"\1", regex=True)
        .str.strip()
    )
    return df

def clean_occupation_labels(df: pd.DataFrame) -> pd.DataFrame:
    column = "Occupation (current) (10 categories)"
    df[column] = (
        df[column]
        .str.replace(r"^\d+\.\s*", "", regex=True)
        .str.strip()
    )
    return df


# Mapping from filename to transformer with correct category column
UK_TRANSFORMERS = {
    "sex.csv": UKCensusTransformer(category_columns=["Sex (2 categories)"]),
    "age_group.csv": UKCensusTransformer(
        category_columns=["Age (B) (18 categories)", "Sex (2 categories)"],
        rename_func=clean_age_labels
    ),
    "household_composition.csv": UKCensusTransformer(
        category_columns=["Household composition (8 categories)"],
        drop_rows=lambda df: df["Household composition (8 categories)"] != "Does not apply",
        rename_func=clean_household_composition_labels
    ),
    "household_size.csv": UKCensusTransformer(
        category_columns=["Household size (9 categories)"],
        rename_func=clean_household_size_labels
    ),
    "occupation.csv": UKCensusTransformer(category_columns=["Occupation (current) (10 categories) Code"]),
}

UN_TRANSFORMERS = {
    "household_size.csv": UNHouseholdTransformer(),
    "household_composition.csv": UNHouseholdTransformer(),
}


def process_uk_location(location_dir: Path):
    print(f"\n📍 Processing UK location: {location_dir.name}")
    loader = UKCensusLoader(location_dir)
    location_output_dir = OUTPUT_DIR / location_dir.name

    for filename, transformer in UK_TRANSFORMERS.items():
        try:
            df_raw = loader.load_file(filename)
            df_processed = transformer.transform(df_raw)
            save_processed(df_processed, location_output_dir, filename)
            print(f"✅ Processed {filename}")
        except FileNotFoundError:
            print(f"⚠️  Missing file: {filename}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")

def process_un_global(global_dir: Path):
    print("\n🌍 Processing UN Global data...")
    xlsx_path = global_dir / "hh_size_composition.xlsx"
    loader = UNHouseholdLoader(str(xlsx_path))

    for filename, transformer in UN_TRANSFORMERS.items():
        try:
            df_raw = loader.load_file(filename)  # This will return country-wide table
            for country in df_raw["Country"].unique():
                country_df = df_raw[df_raw["Country"] == country]
                df_processed = transformer.transform(country_df)
                country_dir = OUTPUT_DIR / country.lower().replace(" ", "_")
                save_processed(df_processed, country_dir, filename)
                print(f"✅ Processed {filename} for {country}")
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")


def main():
    for location_dir in RAW_DATA_DIR.iterdir():
        if not location_dir.is_dir():
            continue

        if location_dir.name == "global":
            process_un_global(location_dir)
        else:
            process_uk_location(location_dir)

if __name__ == "__main__":
    main()
