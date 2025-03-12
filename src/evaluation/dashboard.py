import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, dash_table
import os
import numpy as np
from glob import glob
import json

###############
# SCRIPT CONFIG
###############

var_name = "population_size"

###############
# LOAD INPUTS
###############

DIR = os.path.join(os.path.dirname(__file__), f"../../data/evaluation/{var_name}")

dataframes = []

metadata_files = glob(os.path.join(DIR, f"estimated_{var_name}_*metadata.json"))
for metadata_file in metadata_files:
    with open(metadata_file, "r") as f:
        metadata = json.load(f)
    
    model_name = metadata.get("model_name", "Unknown Model")
    output_file = metadata.get("ouput_file", "").strip()
    
    if not output_file:
        print(f"[WARNING] No output file found in metadata: {metadata_file}")
        continue

    dataset_path = os.path.join(DIR, output_file)

    if not os.path.exists(dataset_path):
        print(f"[WARNING] Dataset file {output_file} not found. Skipping.")
        continue

    df = pd.read_csv(dataset_path)
    df["Model"] = model_name
    dataframes.append(df)

df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()

###############
# ANALYSIS
###############

response_columns = [col for col in df.columns if "Counts_" in col]
df[response_columns] = df[response_columns].apply(pd.to_numeric, errors="coerce")
df["Counts"] = pd.to_numeric(df["Counts"], errors="coerce")


df["LLM_Mean"] = df[response_columns].mean(axis=1)
df["LLM_Std"] = df[response_columns].std(axis=1)
df["LLM_Range"] = df[response_columns].max(axis=1) - df[response_columns].min(axis=1)
df["Absolute_Error"] = abs(df["LLM_Mean"] - df["Counts"])
df["Percentage_Error"] = np.where(
    df["Counts"] > 0,
    (df["Absolute_Error"] / df["Counts"]) * 100,
    np.nan
)

summary_stats = df.groupby("BUA size classification").agg(
    Mean_Actual=("Counts", "mean"),
    Mean_Predicted=("LLM_Mean", "mean"),
    MAE=("Absolute_Error", "mean"),
    MPE=("Percentage_Error", "mean"),
    Std_Deviation=("Absolute_Error", "std")
)


# Create Dash App
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("LLM Population Analysis Dashboard", style={"textAlign": "center"}),

    # Dropdown to select settlement size class
    html.Label("Select Settlement Size:"),
    dcc.Dropdown(
        id="size-class-dropdown",
        options=[{"label": size, "value": size} for size in df["BUA size classification"].unique()],
        value=df["BUA size classification"].unique()[0],  # Default to first option
        clearable=False
    ),

    html.Br(),

    # Dropdown to select models
    html.Label("Select Model(s):"),
    dcc.Dropdown(
        id="model-dropdown",
        options=[{"label": model, "value": model} for model in df["Model"].unique()],
        value=df["Model"].unique(),  # Default to all models selected
        multi=True
    ),

    html.Br(),

    # Comparative Bar Chart
    dcc.Graph(id="bar-chart"),

    # Summary Statistics Table
    html.H3("Summary Statistics"),
    dash_table.DataTable(
        id="summary-table",
        columns=[{"name": col, "id": col} for col in ["BUA size classification", "Model", "MAE", "MPE"]],
        style_table={"overflowX": "auto"}
    )
])

# Callbacks to update charts based on selection
@app.callback(
    [dash.Output("bar-chart", "figure"),
     dash.Output("summary-table", "data")],
    [dash.Input("size-class-dropdown", "value"),
     dash.Input("model-dropdown", "value")]
)
def update_charts(selected_size, selected_models):
    df_filtered = df[(df["BUA size classification"] == selected_size) & (df["Model"].isin(selected_models))]
    
    # Get unique actual values (ensuring only one row per location)
    df_actual = df_filtered[["BUA name", "Counts"]].drop_duplicates(subset=["BUA name"]).copy()
    df_actual["Model"] = "Actual Census"  # Label it as "Actual"
    df_actual = df_actual.rename(columns={"Counts": "Population"})  # Rename column for consistency

    # Keep only necessary columns for predictions
    df_predicted = df_filtered.rename(columns={"LLM_Mean": "Population"})[["BUA name", "Model", "Population"]]

    # Merge actual and predicted values into a single DataFrame
    df_combined = pd.concat([df_actual, df_predicted], ignore_index=True)

    # Create bar chart with actual & predicted populations in the same plot
    bar_fig = px.bar(
        df_combined,
        x="BUA name",
        y="Population",
        color="Model",  # Distinguish actual vs. LLM models
        barmode="group",  # Group bars per location
        title=f"Actual vs. Predicted Population Comparison ({selected_size})"
    )

    bar_fig.update_layout(xaxis_tickangle=-45)  # Rotate labels for readability

    # Summary Stats
    summary_df = df_filtered.groupby(["BUA size classification", "Model"]).agg(
        MAE=("Absolute_Error", "mean"),
        MPE=("Percentage_Error", "mean")
    ).reset_index()

    return bar_fig, summary_df.to_dict("records")

# Run app
if __name__ == "__main__":
    app.run_server(debug=True)
