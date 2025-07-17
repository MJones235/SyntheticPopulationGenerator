import pandas as pd
from pathlib import Path

def save_processed(df: pd.DataFrame, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / filename, index=False)
