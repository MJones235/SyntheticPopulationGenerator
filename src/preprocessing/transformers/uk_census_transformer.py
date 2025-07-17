from .base_transformer import BaseTransformer
import pandas as pd
from typing import List, Optional, Callable

class UKCensusTransformer(BaseTransformer):
    def __init__(
        self,
        category_columns: List[str],
        drop_rows: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
        rename_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None
    ):
        self.category_columns = category_columns
        self.drop_rows = drop_rows
        self.rename_func = rename_func

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.drop_rows is not None:
            df = df[self.drop_rows(df)]

        if self.rename_func is not None:
            df = self.rename_func(df)

        grouped = df.groupby(self.category_columns, as_index=False)["Observation"].sum()
        total = grouped["Observation"].sum()
        grouped["Percentage"] = (grouped["Observation"] / total * 100).round(1)

        rename_dict = {col: f"Category_{i+1}" for i, col in enumerate(self.category_columns)}
        return grouped.rename(columns=rename_dict)[[*rename_dict.values(), "Percentage"]]
