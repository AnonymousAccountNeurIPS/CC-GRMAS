import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_graph_schema(base_url: str):
    st.header("Graph Schema")
    
    if st.button("Check Graph Schema", key="graph_schema_button"):
        with st.spinner("Retrieving graph schema..."):
            response = make_api_request(base_url, Config.ENDPOINTS["graph_schema"])
            data = display_response(response, "Graph schema retrieved")
            if data:
                st.code(data.get("schema", "No schema available"), language="text")