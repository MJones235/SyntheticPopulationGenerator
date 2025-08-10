import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from src.analysis.diversity_and_validity import compute_generation_validity, compute_household_structure_diversity, compute_individual_diversity
from src.classifiers.household_size.dar_es_salaam import DarEsSalaamHouseholdSizeClassifier
from src.classifiers.household_size.uk_census import UKHouseholdSizeClassifier
from src.classifiers.household_size.un_global import UNHouseholdSizeClassifier
from src.classifiers.household_type.uk_census import UKHouseholdCompositionClassifier
from src.classifiers.household_type.un_global import UNHouseholdCompositionClassifier
from src.services.experiment_run_service import ExperimentRunService
from src.services.experiments_service import ExperimentService
from src.analysis.distributions import compute_occupation_distribution
from src.services.population_service import PopulationService
from src.services.metadata_service import MetadataService
from src.services.file_service import FileService
from src.utils.colour_generator import assign_household_colors
from src.utils.plots import plot_age_pyramid, plot_household_size, plot_household_structure_bar, plot_occupation_titles, plot_occupations, plot_age_diff
import os
import altair as alt

from src.analysis.similarity_metrics import compute_aggregate_metrics, compute_convergence_curve, compute_similarity_metrics
from src.utils.aggregate_plots import plot_age_pyramid_aggregate, plot_household_size_aggregate, plot_household_structure_bar_aggregate, plot_occupations_aggregate

def get_household_size_classifier(experiment):
    if experiment['hh_size_classifier'] == 'un_global':
        return UNHouseholdSizeClassifier()
    elif experiment['hh_size_classifier'] == 'dar_es_salaam':
        return DarEsSalaamHouseholdSizeClassifier()
    else:
        return UKHouseholdSizeClassifier()


st.set_page_config(layout="wide")

file_service = FileService()
metadata_service = MetadataService()
population_service = PopulationService()
experiment_service = ExperimentService()
experiment_runs_service = ExperimentRunService()

st.title("📊 Population Browser")

experiments = experiment_service.get()

if not experiments:
    st.warning("No experiments found in the database.")
else:
    experiment_dict = {f"{p['timestamp']} - {p['model']} - {p['location']} - {'stats, ' if bool(p['include_stats']) else ''}{'guidance, ' if bool(p['include_guidance']) else ''}{'target, ' if bool(p['include_target']) else ''}{'microdata, ' if bool(p['use_microdata']) else ''}{'fixed household size, ' if bool(p['compute_household_size']) else ''}{'no occupation, ' if bool(p['no_occupation']) else ''} {'no household composition, ' if bool(p['no_household_composition']) else ''}": p['experiment_id'] for p in experiments}
    selected_exp_label = st.selectbox("Select an Experiment:", list(experiment_dict.keys()))
    selected_experiment_id = experiment_dict[selected_exp_label]
    selected_experiment = experiment_service.get_by_id(selected_experiment_id)
    hh_type_classifier = UNHouseholdCompositionClassifier() if selected_experiment["hh_type_classifier"] == "un_global" else UKHouseholdCompositionClassifier()
    hh_size_classifier = get_household_size_classifier(selected_experiment)
    location = selected_experiment["location"].replace(" ", "_")
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

            # Load all populations
            population_dfs = []
            hh_size_distributions = []
            occupation_distributions = []
            for pid in population_ids:
                try:
                    df = pd.DataFrame(population_service.get_by_id(pid))
                    population_dfs.append(df)
                    hh_size_distributions.append(hh_size_classifier.compute_observed_distribution(df))
                    if not selected_experiment['no_occupation']: occupation_distributions.append(compute_occupation_distribution(df))
                except Exception as e:
                    st.warning(f"Failed to load run {pid}: {e}")

            # Show metrics
            if population_dfs:
                agg_metrics = compute_aggregate_metrics(population_dfs, location, not selected_experiment['no_occupation'], hh_type_classifier, hh_size_classifier)
                st.dataframe(agg_metrics)

                census_household = file_service.load_household_size(location)
                st.pyplot(plot_household_size_aggregate(hh_size_distributions, census_household))
                census_age_df = file_service.load_age_pyramid(location)
                st.pyplot(plot_age_pyramid_aggregate(population_dfs, census_age_df))
                if not selected_experiment['no_occupation']:
                    census_occupation = file_service.load_occupation_distribution(location)
                    st.pyplot(plot_occupations_aggregate(occupation_distributions, census_occupation))
                census_composition_df = file_service.load_household_composition(location)
                st.pyplot(plot_household_structure_bar_aggregate(population_dfs, census_composition_df, hh_type_classifier))
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
                    results = compute_similarity_metrics(df, location, not metadata['no_occupation'], hh_type_classifier, hh_size_classifier)
                    st.dataframe(results)
                except Exception as e:
                    st.error(f"Failed to compute similarity metrics: {e}")

            if selected_experiment.get("include_avg_household_size", False):
                st.title("Average Household Size")

                try:
                    census_avg = file_service.load_avg_household_size(location)
                    synthetic_avg = hh_size_classifier.compute_average_household_size(df)

                    percentage_error = 100 * abs(synthetic_avg - census_avg) / census_avg

                    st.markdown(f"**Census Average:** {census_avg:.2f}")
                    st.markdown(f"**Synthetic Average:** {synthetic_avg:.2f}")
                    st.markdown(f"**Percentage Error:** {percentage_error:.1f}%")
                except Exception as e:
                    st.error(f"Failed to compute average household size: {e}")

            st.title("Diversity")

            try:
                diversity_cols = ["age", "gender", "relationship"]
                if not metadata.get("no_occupation", False):
                    diversity_cols.append("occupation")
                
                individual_diversity = compute_individual_diversity(df, diversity_cols)
                household_diversity = compute_household_structure_diversity(df)
                st.markdown(f"**Individual Diversity Score:** {individual_diversity:.1f}%")
                st.markdown(f"**Household Diversity Score:** {household_diversity:.1f}%")

            except Exception as e:
                st.error(f"Failed to compute diversity score: {e}")


            st.title("Generation Validity")

            try:
                validity = compute_generation_validity(df, metadata["num_households"])
                st.markdown(f"**Validity Score:** {validity:.1f}%")
            except Exception as e:
                st.error(f"Failed to compute generation validity: {e}")


            st.title("Household Size Comparison")
            st.pyplot(plot_household_size(hh_size_classifier.compute_observed_distribution(df), file_service.load_household_size(location)))
        

            st.title("Age Distribution Comparison")
            if not df.empty:
                try:
                    census_age_df = file_service.load_age_pyramid(location)
                    st.pyplot(plot_age_pyramid(df, census_age_df))
                except Exception as e:
                    st.error(f"Failed to load or plot age pyramid: {e}")

            if not metadata['no_occupation']:
                st.title("Occupation Comparison")
                if not df.empty:
                    try:
                        census_occupation = file_service.load_occupation_distribution(location)
                        synthetic_occupation = compute_occupation_distribution(df)
                        st.pyplot(plot_occupations(synthetic_occupation, census_occupation))
                        st.pyplot(plot_occupation_titles(df))
                    except Exception as e:
                        st.error(f"Failed to load or plot occupation: {e}")

            st.title("Parent-Child Age Difference")
            if not df.empty:
                try:
                    fig, mean_mother_birth_age, mean_father_birth_age, median_mother_first_birth_age = plot_age_diff(df)
                    st.pyplot(fig)
                    st.markdown(f"**Mean mother birth age:** {mean_mother_birth_age:.1f}")
                    st.markdown(f"**Mean father birth age:** {mean_father_birth_age:.1f}")
                    st.markdown(f"**Median mother age at first birth:** {median_mother_first_birth_age:.1f}")

                except Exception as e:
                    st.error(f"Failed to load or plot age diff: {e}")

            st.title("Household Composition")
            if not df.empty:
                try:
                    census_composition_df = file_service.load_household_composition(location)
                    st.pyplot(plot_household_structure_bar(df, census_composition_df, hh_type_classifier))
                except Exception as e:
                    st.error(f"Failed to load or plot household composition: {e}")

            if not df.empty:
                try:
                    convergence_df = compute_convergence_curve(df, location, 20, 10000, not metadata["no_occupation"], hh_type_classifier, hh_size_classifier)
                    st.title("Convergence Curve (JSD)")
                    convergence_long_jsd = convergence_df.melt(
                        id_vars=["Variable", "n_individuals"],
                        value_vars=["JSD"], 
                        var_name="Metric",
                        value_name="Value"
                    )
                    chart_jsd = alt.Chart(convergence_long_jsd).mark_line().encode(
                        x="n_individuals:Q",
                        y="Value:Q",
                        color="Variable:N"
                    ).properties(width=750, height=400)

                    st.altair_chart(chart_jsd, use_container_width=True)

                    st.title("Convergence Curve (MaxAbsError)")
                    convergence_long_max_abs_err = convergence_df.melt(
                        id_vars=["Variable", "n_individuals"],
                        value_vars=["MaxAbsError"], 
                        var_name="Metric",
                        value_name="Value"
                    )
                    chart_max_abs_err = alt.Chart(convergence_long_max_abs_err).mark_line().encode(
                        x="n_individuals:Q",
                        y="Value:Q",
                        color="Variable:N"
                    ).properties(width=750, height=400)

                    st.altair_chart(chart_max_abs_err, use_container_width=True)


                except Exception as e:
                    st.error(f"Failed to compute or plot convergence curve: {e}")