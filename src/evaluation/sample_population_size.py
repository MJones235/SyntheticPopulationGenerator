import pandas as pd
import os

from src.services.file_service import FileService

file_service = FileService()
DIR = os.path.join(os.path.dirname(__file__), "../../data/evaluation/population_size")
filepath = os.path.join(DIR, "population_size.csv")

if not os.path.exists(filepath):
    raise FileNotFoundError(f"File not found: {filepath}")

df = pd.read_csv(filepath)
df.columns = df.columns.str.strip()

sample_size = df["BUA size classification"].value_counts().min()
categories = df["BUA size classification"].unique()
print(f"[INFO] Sampling {sample_size} locations from each classification in {categories}")

sampled_dfs = []

for category in categories:
    df_category = df[df["BUA size classification"] == category]
    df_sample = df_category.sample(n=sample_size, replace=False)
    sampled_dfs.append(df_sample)

df_sampled = pd.concat(sampled_dfs)
df_sampled = df_sampled[["Country", "Region", "BUA name", "BUA size classification", "Counts"]]
df_sampled["Counts"] = (
        df_sampled["Counts"]
        .astype(str) 
        .str.replace(",", "", regex=True) 
        .str.strip() 
        .apply(lambda x: int(x) if x.isdigit() else 0)
    )

output_filepath = file_service.generate_unique_filename(DIR, "sampled_population_size.csv")
df_sampled.to_csv(output_filepath, index=False)
print(f"[INFO] Data saved to {output_filepath}")
