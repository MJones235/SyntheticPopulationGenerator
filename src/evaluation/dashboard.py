from src.evaluation.metrics_calculator import MetricsCalculator
from src.repositories.dashboard_repository import DashboardRepository
import pandas as pd
import dash
from dash import dcc, html, dash_table
import pandas as pd
import plotly.express as px

DB_PATH = "outputs.sqlite"
repo = DashboardRepository()

def load_data(variable: str) -> pd.DataFrame:
    return repo.get_estimations_with_metadata(variable)

app = dash.Dash(__name__)
server = app.server 

VARIABLE_OPTIONS = ["population_size"]

app.layout = html.Div([
    html.H1("LLM Estimation Dashboard", style={"textAlign": "center"}),

    html.Label("Select Variable:"),
    dcc.Dropdown(
        id="variable-dropdown",
        options=[{"label": v, "value": v} for v in VARIABLE_OPTIONS],
        value=VARIABLE_OPTIONS[0],
        clearable=False
    ),

    html.Br(),

    html.Label("Select BUA Size:"),
    dcc.Dropdown(id="category-dropdown", multi=False),

    html.Br(),

    html.Label("Select Model(s):"),
    dcc.Dropdown(id="model-dropdown", multi=True),

    html.Br(),

    dcc.Graph(id="bar-chart"),

    html.H3("Summary Statistics"),
    dash_table.DataTable(
        id="summary-table",
        style_table={"overflowX": "auto"}
    )
])

@app.callback(
    [
        dash.Output("category-dropdown", "options"),
        dash.Output("category-dropdown", "value"),
        dash.Output("model-dropdown", "options"),
        dash.Output("model-dropdown", "value")
    ],
    [dash.Input("variable-dropdown", "value")]
)
def update_dropdowns(variable):
    df = load_data(variable)
    categories = df["BUA size classification"].dropna().unique()
    models = sorted(df["model_name"].dropna().unique())

    category_options = [{"label": c, "value": c} for c in categories]
    model_options = [{"label": m, "value": m} for m in models]

    return category_options, categories, model_options, models

@app.callback(
    [
        dash.Output("bar-chart", "figure"),
        dash.Output("summary-table", "data"),
        dash.Output("summary-table", "columns")
    ],
    [
        dash.Input("variable-dropdown", "value"),
        dash.Input("category-dropdown", "value"),
        dash.Input("model-dropdown", "value")
    ]
)
def update_dashboard(variable, selected_categories, selected_models):
    df = load_data(variable)

    df = df[
        df["BUA size classification"].isin([selected_categories])
        & df["model_name"].isin(selected_models)
    ]

    df_grouped = (
        df.groupby(["location", "model_name", "BUA size classification"])
        .agg(ground_truth=("ground_truth", "first"), prediction=("prediction", "median"))
        .reset_index()
    )

    bar_fig = px.bar(
        df_grouped,
        x="location",
        y="prediction",
        color="model_name",
        barmode="group",
        title="Median Predicted vs Actual Population",
    )

    bar_fig.add_scatter(
        x=df_grouped["location"],
        y=df_grouped["ground_truth"],
        mode="markers",
        marker=dict(color="black", symbol="x"),
        name="Actual (Census)",
    )

    mc = MetricsCalculator(df_grouped)
    summary = mc.summary_by_group(group_cols=["BUA size classification", "model_name"])

    columns = [{"name": col, "id": col} for col in summary.columns]
    return bar_fig, summary.to_dict("records"), columns

if __name__ == "__main__":
    app.run_server(debug=True)
