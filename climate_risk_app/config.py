import streamlit as st
from typing import Dict

class Config:
    DEFAULT_CSV_PATH = "data/landslides.csv"
    
    ENDPOINTS = {
        "generate_graph": "/generate_graph",
        "graph_schema": "/graph_schema",
        "create_index": "/create_index",
        "populate_vectorstore": "/populate_vectorstore",
        "similarity_search": "/similarity_search",
        "ask_question": "/ask_question",
        "hybrid_search2": "/hybrid_search",
        "risk_analysis": "/risk_analysis",
        "climate_report": "/climate_report",
        "train": "/train",
        "predict": "/predict",
        "hotspots": "/hotspots",
        "model_status": "/model/status",
        "events_spatial": "/events/spatial",
        "risk_distribution": "/analytics/risk-distribution"
    }

def setup_page_config():
    st.set_page_config(
        page_title="Climate Change Risk Management System",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )