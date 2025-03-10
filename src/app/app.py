import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from src.repositories.population_repository import PopulationRepository
from src.repositories.metadata_repository import MetadataRepository
from src.utils.colour_generator import assign_household_colors
import os

st.set_page_config(layout="wide")

def load_html_report(report_path):
    """Reads and returns the HTML content of a report."""
    try:
        with open(report_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return None


metadata_repo = MetadataRepository()
population_repo = PopulationRepository()

st.title("📊 Population Browser")

populations = metadata_repo.get_all_populations()

if not populations:
    st.warning("No populations found in the database.")
else:
    # Create dropdown menu with timestamps as labels
    population_dict = {f"{p['timestamp']} - {p['location']} - {p['num_households']} households": p['population_id'] for p in populations}
    selected_label = st.selectbox("Select a Population:", list(population_dict.keys()))
    selected_population_id = population_dict[selected_label]

    tab1, tab2, tab3 = st.tabs(["Metadata & population table", "Exploratory data analysis", "Comparison"])

    with tab1:

        # Fetch and display metadata
        metadata = metadata_repo.get_metadata_by_population_id(selected_population_id)
        if metadata:
            st.subheader("📑 Population Metadata")
            st.json(metadata, expanded=False)

        # Fetch and display population data
        individuals = population_repo.get_population_by_id(selected_population_id)

        if not individuals:
            st.warning("No data found for this population.")
        else:
            df = pd.DataFrame(individuals)
            styled_df = assign_household_colors(df)
            st.subheader("👥 Population Data")
            st.write(f"Displaying **{len(df)} individuals** from **Population ID: {selected_population_id}**")
            st.dataframe(styled_df, column_order=('name', 'age', 'gender', 'occupation', 'relationship'))

    with tab2:
        report_path = os.path.join("reports", f"{selected_population_id}.html")
        report_html = load_html_report(report_path)
        components.html(report_html, height=600, scrolling=True)

    with tab3:
        st.text(selected_population_id)