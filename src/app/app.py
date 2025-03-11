import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from src.services.population_service import PopulationService
from src.services.metadata_service import MetadataService
from src.services.file_service import FileService
from src.utils.colour_generator import assign_household_colors
import os

st.set_page_config(layout="wide")

file_service = FileService()
metadata_service = MetadataService()
population_service = PopulationService()

st.title("📊 Population Browser")

populations = metadata_service.get()

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
            st.dataframe(styled_df, column_order=('name', 'age', 'gender', 'occupation', 'relationship'))

    with tab2:
        report_path = os.path.join("reports", f"{selected_population_id}.html")
        report_html = file_service.load_html_report(report_path)
        components.html(report_html, height=600, scrolling=True)

    with tab3:
        st.text(selected_population_id)