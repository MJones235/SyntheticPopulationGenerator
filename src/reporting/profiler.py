import os
from datetime import datetime
import pandas as pd
from ydata_profiling import ProfileReport

def generate_report(df: pd.DataFrame) -> str:
    """Creates a profiling report and returns the filename."""
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = os.path.join(reports_dir, f"synthetic_population_report_{timestamp}.html")

    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["first_name"] = df["name"].apply(lambda x: x.split()[0] if " " in x else x)
    df["surname"] = df["name"].apply(lambda x: x.split()[-1] if " " in x else x)

    profile = ProfileReport(df, title="Synthetic Population Analysis", explorative=True)
    profile.to_file(report_filename)
    return report_filename
