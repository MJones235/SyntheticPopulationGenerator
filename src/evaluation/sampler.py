import pandas as pd
import os
from src.services.file_service import FileService

class Sampler:
    def __init__(self, variable: str):
        self.variable = variable
        self.file_service = FileService()
        self.input_path = f"data/evaluation/{variable}/{variable}.csv"
        self.output_dir = f"data/evaluation/{variable}"

    def sample(self) -> str:
        if not os.path.exists(self.input_path):
            raise FileNotFoundError(f"File not found: {self.input_path}")

        df = pd.read_csv(self.input_path)
        df.columns = df.columns.str.strip()

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
