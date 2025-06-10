import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from src.services.experiment_run_service import ExperimentRunService
from src.services.experiments_service import ExperimentService
from src.analysis.distributions import compute_occupation_distribution
from src.services.analysis_service import AnalysisService
from src.services.population_service import PopulationService
from src.services.metadata_service import MetadataService
from src.services.file_service import FileService
from src.utils.colour_generator import assign_household_colors
from src.utils.plots import plot_age_pyramid, plot_household_size, plot_household_structure_bar, plot_occupation_titles, plot_occupations, plot_age_diff
import os

from src.analysis.similarity_metrics import compute_aggregate_metrics, compute_similarity_metrics
from src.utils.aggregate_plots import plot_household_size_aggregate

st.set_page_config(layout="wide")

file_service = FileService()
metadata_service = MetadataService()
population_service = PopulationService()
analysis_service = AnalysisService()
experiment_service = ExperimentService()
experiment_runs_service = ExperimentRunService()

st.title("📊 Population Browser")

experiments = experiment_service.get()

if not experiments:
    st.warning("No experiments found in the database.")
else:
    experiment_dict = {f"{p['timestamp']} - {p['model']} - {p['location']} - {'stats, ' if bool(p['include_stats']) else ''}{'guidance, ' if bool(p['include_guidance']) else ''}{'target, ' if bool(p['include_target']) else ''}{'microdata, ' if bool(p['use_microdata']) else ''}{'fixed household size, ' if bool(p['compute_household_size']) else ''}{'no occupation, ' if bool(p['no_occupation']) else ''}": p['experiment_id'] for p in experiments}
    selected_exp_label = st.selectbox("Select an Experiment:", list(experiment_dict.keys()))
    selected_experiment_id = experiment_dict[selected_exp_label]

    runs = experiment_runs_service.get_by_experiment_id(selected_experiment_id)

    if not runs:
        st.warning("No runs found for this experiment")

    else:
        run_options = {f"Run {r['run_number']} ({r['timestamp']})": r['population_id'] for r in runs}
        selected_run_label = st.selectbox("Select a Specific Run:", list(run_options.keys()))
        selected_population_id = run_options[selected_run_label]

        aggregate_tab, tab1, tab2, tab3 = st.tabs(["Aggregate Stats", "Metadata & population table", "Exploratory data analysis", "Comparison"])

        with aggregate_tab:
            st.subheader("📊 Aggregate Metrics for Experiment")

            population_ids = [r["population_id"] for r in runs]
            location = experiments[0]["location"]

            # Load all populations
            population_dfs = []
            hh_size_distributions = []
            for pid in population_ids:
                try:
                    df = pd.DataFrame(population_service.get_by_id(pid))
                    population_dfs.append(df)
                    analysis = analysis_service.get_by_id(pid)
                    hh_size_distributions.append(analysis["household_size_distribution"])
                except Exception as e:
                    st.warning(f"Failed to load run {pid}: {e}")

            # Show metrics
            if population_dfs:
                agg_metrics = compute_aggregate_metrics(population_dfs, location)
                st.dataframe(agg_metrics)

                census_household = file_service.load_household_size(location)
                st.pyplot(plot_household_size_aggregate(hh_size_distributions, census_household))

                # Optional: show age/occupation/household composition graphs with CI
                # st.pyplot(plot_age_pyramid_aggregate(population_dfs, census_age_df))
                # st.pyplot(plot_occupations_aggregate(...))
                # st.pyplot(plot_household_structure_bar_aggregate(...))
            else:
                st.warning("No valid population data available for aggregate analysis.")

        with tab1:

            # Fetch and display metadata
            metadata = metadata_service.get_by_id(selected_population_id)
            if metadata:
                st.subheader("📑 Population Metadata")
                st.json(metadata, expanded=False)

            # Fetch and display population data
            individuals = population_service.get_by_id(selected_population_id)

            if not individuals:
                st.warning("No data found for this population.")
            else:
                df = pd.DataFrame(individuals)
                styled_df = assign_household_colors(df)
                st.subheader("👥 Population Data")
                st.write(f"Displaying **{len(df)} individuals** from **Population ID: {selected_population_id}**")
                st.dataframe(styled_df, column_order=('name', 'age', 'gender', 'occupation_category', 'occupation', 'relationship'))

        with tab2:
            report_path = os.path.join("reports", f"{selected_population_id}.html")
            report_html = file_service.load_html_report(report_path)
            components.html(report_html, height=600, scrolling=True)

        with tab3:
            st.title("Metrics")
            if not df.empty:
                try:
                    results = compute_similarity_metrics(df, metadata["location"])
                    st.dataframe(results)
                except Exception as e:
                    st.error(f"Failed to compute similarity metrics: {e}")
            
            st.title("Household Size Comparison")
            analysis = analysis_service.get_by_id(selected_population_id)
            st.pyplot(plot_household_size(analysis['household_size_distribution'], file_service.load_household_size(metadata['location'])))

            st.title("Age Distribution Comparison")
            if not df.empty:
                try:
                    census_age_df = file_service.load_age_pyramid(metadata["location"])
                    st.pyplot(plot_age_pyramid(df, census_age_df))
                except Exception as e:
                    st.error(f"Failed to load or plot age pyramid: {e}")

            st.title("Occupation Comparison")
            if not df.empty:
                try:
                    census_occupation = file_service.load_occupation_distribution(metadata["location"])
                    synthetic_occupation = compute_occupation_distribution(df)
                    st.pyplot(plot_occupations(synthetic_occupation, census_occupation))
                    st.pyplot(plot_occupation_titles(df))
                except Exception as e:
                    st.error(f"Failed to load or plot occupation: {e}")

            st.title("Parent-Child Age Difference")
            if not df.empty:
                try:
                    st.pyplot(plot_age_diff(df))
                except Exception as e:
                    st.error(f"Failed to load or plot age diff: {e}")

            st.title("Household Composition")
            if not df.empty:
                try:
                    census_composition_df = file_service.load_household_composition(metadata["location"])
                    st.pyplot(plot_household_structure_bar(df, census_composition_df))
                except Exception as e:
                    st.error(f"Failed to load or plot household composition: {e}")
