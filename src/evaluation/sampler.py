import pandas as pd
import os
from src.services.file_service import FileService

class Sampler:

    def __init__(self, variable: str):
        self.variable = variable
        self.file_service = FileService()
        self.input_path = f"data/evaluation/{variable}/{variable}.csv"
        self.output_dir = f"data/evaluation/{variable}"

    def sample(self, mode: str = "sample") -> str:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        df.columns = df.columns.str.strip()

        if mode == "sample":
            return self._sample_by_bua_category(df)
        elif mode == "distribution":
            return self._generate_age_distribution(df)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'sample' or 'distribution'.")

    def _sample_by_bua_category(self, df: pd.DataFrame) -> str:
        sample_size = df["BUA size classification"].value_counts().min()
        categories = df["BUA size classification"].unique()
        print(f"[INFO] Sampling {sample_size} locations from each of {categories}")

        sampled_dfs = []
        for category in categories:
            df_cat = df[df["BUA size classification"] == category]
            df_sample = df_cat.sample(n=sample_size, replace=False)
            sampled_dfs.append(df_sample)

        df_sampled = pd.concat(sampled_dfs)
        df_sampled = df_sampled[["Country", "Region", "BUA name", "BUA size classification", "Counts"]]
        df_sampled["Counts"] = (
            df_sampled["Counts"]
            .astype(str)
            .str.replace(",", "")
            .str.strip()
            .apply(lambda x: int(x) if x.isdigit() else 0)
        )

        output_path = self.file_service.generate_unique_filename(self.output_dir, f"sampled_{self.variable}.csv")
        df_sampled.to_csv(output_path, index=False)
        print(f"[INFO] Sample saved to {output_path}")
        return output_path

    def _generate_age_distribution(self, df: pd.DataFrame) -> str:
        df = df[df["BUA size classification"] == "Major"]

        df["Percentage per BUA"] = pd.to_numeric(
            df["Percentage per BUA"]
            .astype(str)
            .str.replace("%", "")
            .str.strip(),
            errors="coerce"
        ).fillna(0.0)

        pivot_df = df.pivot_table(
            index=["BUA name", "Region", "Country", "BUA size classification"],
            columns=["Age", "Sex"],
            values="Percentage per BUA"
        )

        # Flatten MultiIndex column names
        pivot_df.columns = [
            f"{col[0]} {col[1]}".strip() if isinstance(col, tuple) and col[1] else col[0] if isinstance(col, tuple) else col
            for col in pivot_df.columns
        ]

        pivot_df = pivot_df.reset_index()
        pivot_df.columns = [str(col).strip() for col in pivot_df.columns]

        # Reorder to put BUA size classification after Country
        location_cols = ["BUA name", "Region", "Country", "BUA size classification"]
        other_cols = [col for col in pivot_df.columns if col not in location_cols]
        pivot_df = pivot_df[location_cols + other_cols]

        output_path = self.file_service.generate_unique_filename(self.output_dir, "sampled_age_distribution.csv")
        pivot_df.to_csv(output_path, index=False)
        print(f"[INFO] Age distribution saved to {output_path}")
        return output_path