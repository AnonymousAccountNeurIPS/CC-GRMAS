import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_generate_graph(base_url: str):
    st.header("Generate Graph")
    
    csv_path = st.text_input(
        "CSV File Path",
        value=Config.DEFAULT_CSV_PATH,
        placeholder="Enter the path to your landslide CSV file",
        key="generate_graph_csv_path"
    )
    
    clear_existing = st.checkbox(
        "Clear existing data",
        value=False,
        key="generate_graph_clear_existing"
    )
    
    if st.button("Generate Graph", key="generate_graph_button"):
        if csv_path:
            with st.spinner("Creating graph from CSV data..."):
                response = make_api_request(
                    base_url,
                    Config.ENDPOINTS["generate_graph"],
                    "POST",
                    {"csv_path": csv_path, "clear_existing": clear_existing}
                )
                data = display_response(response, "Graph created successfully!")
                
                if data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Records Processed", data.get("records_processed", 0))
                    with col2:
                        st.metric("Records Failed", data.get("records_failed", 0))
                    with col3:
                        st.metric("Total Records", data.get("total_records", 0))
        else:
            st.error("Please enter a CSV file path")