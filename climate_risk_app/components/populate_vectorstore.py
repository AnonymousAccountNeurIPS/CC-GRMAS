import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_populate_vectorstore(base_url: str):
    st.header("Populate Vector Store")
    
    if st.button("Populate Vector Store", key="populate_vectorstore_button"):
        with st.spinner("Populating vector store with embeddings..."):
            response = make_api_request(base_url, Config.ENDPOINTS["populate_vectorstore"], "POST")
            data = display_response(response, "Vector store populated successfully!")
            if data:
                st.metric("Documents Processed", data.get("processed_count", 0))