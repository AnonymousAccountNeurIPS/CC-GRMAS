import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_create_index(base_url: str):
    st.header("Create Vector Index")
    
    if st.button("Create Vector Index", key="create_index_button"):
        with st.spinner("Creating vector index..."):
            response = make_api_request(base_url, Config.ENDPOINTS["create_index"], "POST")
            display_response(response, "Vector index created successfully!")