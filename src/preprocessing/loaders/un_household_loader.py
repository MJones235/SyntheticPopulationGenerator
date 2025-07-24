from .base_loader import BaseLoader
import pandas as pd
from typing import List


class UNHouseholdLoader(BaseLoader):
    def __init__(self, data_path: str):
        """
        Load UN household data from a single Excel file.
        """
        super().__init__(location_dir=None)
        self.data_path = data_path
        self.sheet_name = "HH size and composition 2022"
        self.df = self._load_excel()

    def _load_excel(self) -> pd.DataFrame:
        df = pd.read_excel(
            self.data_path,
            sheet_name=self.sheet_name,
            header=[3, 4] 
        )

        # Flatten column headers: skip any NaN, join parts, strip
        df.columns = [
            self._clean_column_name(" ".join(str(c).strip() for c in col if pd.notna(c)))
            for col in df.columns
        ]


        df = df.replace("..", pd.NA)
        df = df.dropna(axis=1, how="all")

        # Identify the reference date column by suffix
        ref_col = next(
            (col for col in df.columns if col.endswith("Reference date (dd/mm/yyyy)")),
            None
        )

        if ref_col:
            df[ref_col] = pd.to_datetime(df[ref_col], errors="coerce", dayfirst=True)
            df = df.rename(columns={ref_col: "Reference date (dd/mm/yyyy)"})
        else:
            raise KeyError("Could not find 'Reference date (dd/mm/yyyy)' column after flattening.")
        
        country_col = next(
            (col for col in df.columns if col.endswith("Country or area")),
            None
        )
        if country_col:
            df = df.rename(columns={country_col: "Country or area"})
        else:
            raise KeyError("Could not find 'Country or area' column after flattening.")

        return df
    
    def _clean_column_name(self, name: str) -> str:
        return name.replace("\n", " ").replace("  ", " ").strip()

    def load_file(self, filename: str) -> pd.DataFrame:
        if filename == "household_size.csv":
            required_columns = list(self.HOUSEHOLD_SIZE_COLUMNS.keys())
            mapping = self.HOUSEHOLD_SIZE_COLUMNS
            return self._extract_latest_entries(required_columns, mapping)

        elif filename == "household_composition.csv":
            required_columns = list(self.HOUSEHOLD_COMPOSITION_COLUMNS.keys())
            mapping = self.HOUSEHOLD_COMPOSITION_COLUMNS
            return self._extract_latest_entries(required_columns, mapping)

        elif filename == "avg_household_size.csv":
            return self._extract_avg_household_size()

        else:
            raise ValueError(f"Unsupported file type: {filename}")


    def _extract_latest_entries(self, required_columns: List[str], mapping: dict) -> pd.DataFrame:
        # Keep only rows where all required columns are present
        valid = self.df.dropna(subset=required_columns + ["Country or area", "Reference date (dd/mm/yyyy)"])

        # Keep the most recent per country
        most_recent = (
            valid.sort_values("Reference date (dd/mm/yyyy)", ascending=False)
            .groupby("Country or area", as_index=False)
            .first()
        )

        # Create rows: Country | Category_1 | Percentage
        rows = []
        for _, row in most_recent.iterrows():
            country = row["Country or area"]
            for col, label in mapping.items():
                if pd.notna(row[col]):
                    rows.append((country, label, float(row[col])))

        return pd.DataFrame(rows, columns=["Country", "Category_1", "Percentage"])
    
    def _extract_avg_household_size(self) -> pd.DataFrame:
        col = "Unnamed: 4_level_0 Average household size (number of members)"
        if col not in self.df.columns:
            raise KeyError(f"Expected column '{col}' not found.")
        
        df_valid = self.df.dropna(subset=["Country or area", "Reference date (dd/mm/yyyy)", col])

        most_recent = (
            df_valid.sort_values("Reference date (dd/mm/yyyy)", ascending=False)
            .groupby("Country or area", as_index=False)
            .first()
        )

        return most_recent[["Country or area", col]].rename(
            columns={
                "Country or area": "Country",
                col: "Value"
            }
        ).assign(Category_1="Average household size")[["Country", "Category_1", "Value"]]


    HOUSEHOLD_SIZE_COLUMNS = {
        "Households by size (percentage ) 1 member": "1",
        "Households by size (percentage ) 2-3 members": "2-3",
        "Households by size (percentage ) 4-5 members": "4-5",
        "Households by size (percentage ) 6 or more members": "6+"
    }

    HOUSEHOLD_COMPOSITION_COLUMNS = {
        "Basic household types (percentage of households)** One-person": "One-person",
        "Basic household types (percentage of households)** Couple only": "Couple",
        "Basic household types (percentage of households)** Couple with children": "Couple with children",
        "Basic household types (percentage of households)** Single parent with children": "Lone parent",
        "Basic household types (percentage of households)** Extended family": "Extended family",
        "Basic household types (percentage of households)** Non-relatives": "Non-relatives",
        "Basic household types (percentage of households)** Unknown": "Unknown"
    }

