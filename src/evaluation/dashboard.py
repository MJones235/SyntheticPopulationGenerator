from src.evaluation.metrics_calculator import MetricsCalculator
from src.repositories.dashboard_repository import DashboardRepository
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
import plotly.express as px

repo = DashboardRepository()

def load_data(variable: str) -> pd.DataFrame:
    return repo.get_estimations_with_metadata(variable)

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server 

VARIABLE_OPTIONS = ["population_size", "age_distribution"]

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

    html.Div(id="dynamic-controls"),

    html.Br(),

    dcc.Graph(id="main-graph"),

    html.H3("Summary Statistics"),
    dash_table.DataTable(
        id="summary-table",
        style_table={"overflowX": "auto"}
    )
])

@app.callback(
    dash.Output("dynamic-controls", "children"),
    [dash.Input("variable-dropdown", "value")]
)
def render_controls(variable):
    df = load_data(variable)
    models = sorted(df["model_name"].dropna().unique())
    model_options = [{"label": m, "value": m} for m in models]

    categories = df["BUA size classification"].dropna().unique()
    category_options = [{"label": c, "value": c} for c in categories]

    locations = sorted(df["location"].dropna().unique())
    location_options = [{"label": l, "value": l} for l in locations]

    return html.Div([
        html.Div([
            html.Label("Select BUA Size:"),
            dcc.Dropdown(id="category-dropdown", options=category_options, value=categories[0])
        ], style={"display": "block" if variable == "population_size" else "none"}),

        html.Div([
            html.Label("Select Location(s):"),
            dcc.Dropdown(
                id="location-dropdown", 
                options=location_options, 
                value=locations[0] if variable == "age_distribution" else locations,
                multi=(variable != "age_distribution")
            )
        ]),

        html.Br(),
        html.Label("Select Model(s):"),
        dcc.Dropdown(
            id="model-dropdown",
            options=model_options,
            value=models[0] if variable == "age_distribution" else models,
            multi=(variable == "population_size")
        ),

        html.Br(),
        html.Div([
            html.Label("Normalised View:"),
            dcc.Checklist(
                id="normalise-toggle",
                options=[{"label": "Show normalised percentages", "value": "normalised"}],
                value=[]
            )
        ], style={"display": "block" if variable == "age_distribution" else "none"})
        
    ])

@app.callback(
    dash.Output("location-dropdown", "options"),
    dash.Output("location-dropdown", "value"),
    dash.Input("category-dropdown", "value"),
    dash.Input("variable-dropdown", "value"),
    prevent_initial_call=True
)
def update_location_options(selected_category, variable):
    if variable != "population_size" or not selected_category:
        return [], []

    df = load_data("population_size")
    filtered = df[df["BUA size classification"] == selected_category]
    locations = sorted(filtered["location"].dropna().unique())
    options = [{"label": loc, "value": loc} for loc in locations]
    return options, locations  # all selected by default


# Shared callback
@app.callback(
    [
        dash.Output("main-graph", "figure"),
        dash.Output("summary-table", "data"),
        dash.Output("summary-table", "columns")
    ],
    [
        dash.Input("variable-dropdown", "value"),
        dash.Input("category-dropdown", "value"),
        dash.Input("location-dropdown", "value"),
        dash.Input("model-dropdown", "value"),
        dash.Input("normalise-toggle", "value")
    ]
)
def update_dashboard(variable, selected_category, selected_location, selected_models, normalise_toggle):
    if selected_models is None:
        return {}, [], []

    df = load_data(variable)
    filtered_df = df.copy()

    if isinstance(selected_models, str):
        selected_models = [selected_models]
    filtered_df = filtered_df[filtered_df["model_name"].isin(selected_models)]

    if variable == "population_size":
        if selected_category is None or not selected_location:
            return {}, [], []
        
        if isinstance(selected_location, str):
            selected_location = [selected_location]

        filtered_df = filtered_df[
            (filtered_df["BUA size classification"] == selected_category) &
            (filtered_df["location"].isin(selected_location))
        ]


        df_grouped = (
            filtered_df.groupby(["location", "model_name", "BUA size classification"])
            .agg(ground_truth=("ground_truth", "first"), prediction=("prediction", "median"))
            .reset_index()
        )

        fig = px.bar(
            df_grouped,
            x="location",
            y="prediction",
            color="model_name",
            barmode="group",
            title="Median Predicted vs Actual Population",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.add_scatter(
            x=df_grouped["location"],
            y=df_grouped["ground_truth"],
            mode="markers",
            marker=dict(color="black", symbol="x"),
            name="Actual (Census)",
        )

        mc = MetricsCalculator(filtered_df)
        summary = mc.summary_by_group(group_cols=["BUA size classification", "model_name"])

    elif variable == "age_distribution":
        if selected_location is None:
            return {}, [], []

        normalise = "normalised" in normalise_toggle
        if normalise:
            filtered_df["prediction"] = (
                filtered_df.groupby(["location", "model_name"])["prediction"]
                .transform(lambda x: x / x.sum() * 100)
            )
            filtered_df["ground_truth"] = (
                filtered_df.groupby(["location"])["ground_truth"]
                .transform(lambda x: x / x.sum() * 100)
            )

        filtered_df = filtered_df[filtered_df["location"] == selected_location].copy()
        filtered_df[["age_band", "sex"]] = filtered_df["subcategory"].str.extract(r"^(.*)\s+(Male|Female)$")

        mc = MetricsCalculator(df)
        
        # Compute both raw and normalised metrics
        raw_metrics = MetricsCalculator(df).summary_by_group(group_cols=["model_name"])        

        norm_metrics = MetricsCalculator(
            df.assign(
                prediction=df.groupby(["location", "model_name"])["prediction"].transform(lambda x: x / x.sum() * 100),
                ground_truth=df.groupby(["location", "model_name"])["ground_truth"].transform(lambda x: x / x.sum() * 100)
            )
        ).summary_by_group(group_cols=["model_name"])

        summary = pd.merge(
            raw_metrics.rename(columns={"MAE": "mae_raw", "MPE": "mpe_raw"}),
            norm_metrics.rename(columns={"MAE": "mae_norm", "MPE": "mpe_norm"}),
            on="model_name"
        )

        fig = go.Figure()
        sexes = ["Male", "Female"]

        def age_band_sort_key(age_band):
            import re
            match = re.search(r"(\d+)", age_band)
            return int(match.group(1)) if match else float("inf")

        age_bands = sorted(filtered_df["age_band"].dropna().unique(), key=age_band_sort_key)

        for model in selected_models:
            for sex in sexes:
                d = filtered_df[(filtered_df["model_name"] == model) & (filtered_df["sex"] == sex)]
                d = d.set_index("age_band").reindex(age_bands).reset_index()
                values = d["prediction"].fillna(0)
                values = -values if sex == "Male" else values
                color = "blue" if sex == "Male" else "red"

                fig.add_bar(
                    y=d["age_band"],
                    x=values,
                    name=f"{model} ({sex})",
                    orientation="h",
                    marker=dict(color=color),
                    opacity=0.6
                )

        for sex in sexes:
            d = filtered_df[filtered_df["sex"] == sex]
            d = d.set_index("age_band").reindex(age_bands).reset_index()
            values = d["ground_truth"].fillna(0)
            values = -values if sex == "Male" else values
            color = "blue" if sex == "Male" else "red"

            fig.add_bar(
                y=d["age_band"],
                x=values,
                name=f"Ground Truth ({sex})",
                orientation="h",
                marker=dict(color=color),
                opacity=0.2
            )

        fig.update_layout(
            title=f"Population Pyramid for {selected_location}",
            barmode="overlay",
            xaxis_title="Percentage of Population",
            yaxis_title="Age Group",
            xaxis_tickformat=".1f"
        )

    columns = [{"name": col, "id": col} for col in summary.columns]
    return fig, summary.to_dict("records"), columns

if __name__ == "__main__":
    app.run(debug=False)