import re
from llm_knowledge_evaluation.core.metrics_calculator import MetricsCalculator
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

VARIABLE_OPTIONS = ["age_distribution", "household_size"]

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
    ),

    html.H3("Distribution Across All Locations by Category and Model"),
    dcc.Graph(id="boxplot-graph"),

])

@app.callback(
    dash.Output("dynamic-controls", "children"),
    [dash.Input("variable-dropdown", "value")]
)
def render_controls(variable):
    df = load_data(variable)
    models = sorted(df["model_name"].dropna().unique())
    model_options = [{"label": m, "value": m} for m in models]

    locations = sorted(df["location"].dropna().unique())
    location_options = [{"label": l, "value": l} for l in locations]

    return html.Div([
        html.Div([
            html.Label("Select Location(s):"),
            dcc.Dropdown(
                id="location-dropdown", 
                options=location_options, 
                value=locations[0] if variable in ["age_distribution", "household_size"] else locations,
            )
        ]),

        html.Br(),
        html.Label("Select Model(s):"),
        dcc.Dropdown(
            id="model-dropdown",
            options=model_options,
            multi=True,
            value=[]
        ),        
    ])

# Shared callback
@app.callback(
    [
        dash.Output("main-graph", "figure"),
        dash.Output("summary-table", "data"),
        dash.Output("summary-table", "columns"),
        dash.Output("boxplot-graph", "figure"),
    ],
    [
        dash.Input("variable-dropdown", "value"),
        dash.Input("location-dropdown", "value"),
        dash.Input("model-dropdown", "value")
    ]
)
def update_dashboard(variable, selected_location, selected_models):
    if selected_models is None:
        return {}, [], [], {}

    df = load_data(variable)
    filtered_df = df.copy()

    if isinstance(selected_models, str):
        selected_models = [selected_models]
    filtered_df = filtered_df[filtered_df["model_name"].isin(selected_models)]


    if variable == "age_distribution":

        if selected_location is None:
            return {}, [], [], {}
        
        model_colors = {model: color for model, color in zip(selected_models, px.colors.qualitative.Set2)}
        filtered_df = filtered_df[filtered_df["location"] == selected_location].copy()
        filtered_df["age_band"] = filtered_df["subcategory"]

        mc = MetricsCalculator(df)
        summary = mc.summary_by_group(group_cols=["model_name"])

        age_bands = sorted(filtered_df["age_band"].dropna().unique(), key=lambda x: int(x.split("-")[0]) if "-" in x else 80)

        fig = go.Figure()

        for model in selected_models:
            d = filtered_df[filtered_df["model_name"] == model]
            d = d.set_index("age_band").reindex(age_bands).reset_index()
            fig.add_bar(
                x=d["age_band"],
                y=d["prediction"],
                name=f"{model} (Prediction)",
                marker=dict(color=model_colors[model])
            )

        d_truth = filtered_df.drop_duplicates(subset=["age_band"])[["age_band", "ground_truth"]].set_index("age_band").reindex(age_bands).reset_index()
        fig.add_bar(
            x=d_truth["age_band"],
            y=d_truth["ground_truth"],
            name="Ground Truth",
            marker_color="black",
            opacity=0.4
        )

        fig.update_layout(
            title=f"Age Distribution in {selected_location}",
            xaxis_title="Age Band",
            yaxis_title="Percentage of Population",
            barmode="group"
        )

        df_long = pd.concat([
            df[["location", "subcategory", "prediction", "model_name"]].rename(columns={"prediction": "value", "model_name": "model"}),
            df[["location", "subcategory", "ground_truth"]].rename(columns={"ground_truth": "value"}).assign(model="Ground Truth")
        ])


        df_long = df_long.dropna(subset=["value"])
        df_long["subcategory"] = df_long["subcategory"].astype(str)
        df_long = df_long[df_long["model"].isin(selected_models) | (df_long["model"] == "Ground Truth")]

        # Create boxplot
        boxplot_fig = px.box(
            df_long,
            x="subcategory",
            y="value",
            color="model",
            color_discrete_sequence=px.colors.qualitative.Set2,
            points="outliers",
            title="Distribution Across Locations by Age Group and Model",
            labels={"value": "Percentage of Population", "subcategory": "Age Group"},
            category_orders={"subcategory": sorted(df_long["subcategory"].unique(), key=lambda x: int(x.split("-")[0]) if "-" in x else 80)}
        )

        boxplot_fig.update_layout(
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2.5,
                minor=dict(ticklen=4, tickcolor="rgba(200,200,200,0.3)", tick0=0, dtick=0.5, showgrid=True)
            ),
            boxmode="group"
        )
    
    elif variable == "household_size":
        if selected_location is None:
            return {}, [], [], {}

        model_colors = {model: color for model, color in zip(selected_models, px.colors.qualitative.Set2)}
        filtered_df = filtered_df[filtered_df["location"] == selected_location].copy()
        filtered_df["household_size"] = (
            filtered_df["subcategory"]
            .str.replace(" in household", "", regex=False)
            .str.strip()
        )


        mc = MetricsCalculator(filtered_df)
        summary = mc.summary_by_group(group_cols=["model_name"])

        size_order = sorted(
            filtered_df["household_size"].dropna().unique(),
            key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 99
        )

        fig = go.Figure()

        for model in selected_models:
            d = filtered_df[filtered_df["model_name"] == model]
            d = d.set_index("household_size").reindex(size_order).reset_index()
            fig.add_bar(
                x=d["household_size"],
                y=d["prediction"],
                name=f"{model} (Prediction)",
                marker=dict(color=model_colors[model])
            )

        d_truth = filtered_df.drop_duplicates(subset=["household_size"])[["household_size", "ground_truth"]].set_index("household_size").reindex(size_order).reset_index()
        fig.add_bar(
            x=d_truth["household_size"],
            y=d_truth["ground_truth"],
            name="Ground Truth",
            marker_color="black",
            opacity=0.4
        )

        fig.update_layout(
            title=f"Household Size Distribution in {selected_location}",
            xaxis_title="Household Size",
            yaxis_title="Percentage of Households",
            barmode="group"
        )       

        df_long = pd.concat([
            df[["location", "subcategory", "prediction", "model_name"]].rename(columns={"prediction": "value", "model_name": "model"}),
            df[["location", "subcategory", "ground_truth"]].rename(columns={"ground_truth": "value"}).assign(model="Ground Truth")
        ])


        df_long = df_long.dropna(subset=["value"])
        df_long["subcategory"] = df_long["subcategory"].astype(str)
        df_long = df_long[df_long["model"].isin(selected_models) | (df_long["model"] == "Ground Truth")]

        # Create boxplot
        boxplot_fig = px.box(
            df_long,
            x="subcategory",
            y="value",
            color="model",
            color_discrete_sequence=px.colors.qualitative.Set2,
            points="outliers",
            title="Distribution Across Locations by Household Size and Model",
            labels={"value": "Percentage of Households", "subcategory": "Household Size"},
            category_orders={"subcategory": sorted(df_long["subcategory"].unique(), key=lambda x: int(x.split("-")[0]) if "-" in x else 80)}
        )

        boxplot_fig.update_layout(
            yaxis=dict(
                tickmode='linear',
                tick0=0,
                dtick=2.5,
                minor=dict(ticklen=4, tickcolor="rgba(200,200,200,0.3)", tick0=0, dtick=0.5, showgrid=True)
            ),
            boxmode="group"
        ) 


    columns = [{"name": col, "id": col} for col in summary.columns]
    return fig, summary.to_dict("records"), columns, boxplot_fig

if __name__ == "__main__":
    app.run(debug=False)