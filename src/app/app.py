import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from src.analysis.distributions import compute_occupation_distribution
from src.services.analysis_service import AnalysisService
from src.services.population_service import PopulationService
from src.services.metadata_service import MetadataService
from src.services.file_service import FileService
from src.utils.colour_generator import assign_household_colors
from src.utils.plots import plot_age_pyramid, plot_household_size, plot_household_structure_bar, plot_occupation_titles, plot_occupations, plot_age_diff
import os

st.set_page_config(layout="wide")

file_service = FileService()
metadata_service = MetadataService()
population_service = PopulationService()
analysis_service = AnalysisService()

st.title("📊 Population Browser")

populations = metadata_service.get()

if not populations:
    st.warning("No populations found in the database.")
else:
    # Create dropdown menu with timestamps as labels
    population_dict = {f"{p['timestamp']} - {p['model']} - {p['num_households']} households - stats {bool(p['include_stats'])} - guidance {bool(p['include_guidance'])}": p['population_id'] for p in populations}
    selected_label = st.selectbox("Select a Population:", list(population_dict.keys()))
    selected_population_id = population_dict[selected_label]

    tab1, tab2, tab3 = st.tabs(["Metadata & population table", "Exploratory data analysis", "Comparison"])

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