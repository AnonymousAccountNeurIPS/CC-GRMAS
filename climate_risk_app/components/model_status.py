import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_model_status(base_url: str):
    st.header("Model Status")
    
    if st.button("Check Model Status", key="model_status_button"):
        with st.spinner("Retrieving model status..."):
            response = make_api_request(base_url, Config.ENDPOINTS["model_status"])
            data = display_response(response, "Model status retrieved")
            if data:
                st.json(data)