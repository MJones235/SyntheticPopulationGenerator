import pandas as pd
import numpy as np

def sample_microdata(microdata_df: pd.DataFrame, n: int) -> pd.DataFrame:
    microdata_df["sampling_weight"] = np.where(
        microdata_df["hh_size_9a"] == 0,
        0,
        1 / microdata_df["hh_size_9a"]
    )
    return microdata_df.sample(
        n=n,
        weights=microdata_df["sampling_weight"],
        replace=False
    )