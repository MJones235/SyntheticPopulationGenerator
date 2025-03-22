import pandas as pd

class MetricsCalculator:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._prepare()

    def _prepare(self):
        self.df = self.df.dropna(subset=["ground_truth", "prediction"])
        self.df["absolute_error"] = (self.df["prediction"] - self.df["ground_truth"]).abs()
        self.df["percentage_error"] = self.df.apply(
            lambda row: (row["absolute_error"] / row["ground_truth"] * 100)
            if row["ground_truth"] != 0 else None,
            axis=1
        )

    def compute_overall_metrics(self):
        return {
            "MAE": self.df["absolute_error"].mean(),
            "MPE": self.df["percentage_error"].mean(),
            "Std_Deviation": self.df["absolute_error"].std(),
        }

    def summary_by_group(self, group_cols=["category", "model_name"]):
        return self.df.groupby(group_cols).agg(
            MAE=("absolute_error", "mean"),
            MPE=("percentage_error", "mean")
        ).reset_index()

    def get_prepared_df(self):
        return self.df
