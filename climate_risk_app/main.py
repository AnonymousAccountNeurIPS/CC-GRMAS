import streamlit as st
from config import Config, setup_page_config
from api_utils import check_connection
from components.home import show_home_page
from components.generate_graph import show_generate_graph
from components.graph_schema import show_graph_schema
from components.create_index import show_create_index
from components.populate_vectorstore import show_populate_vectorstore
from components.similarity_search import show_similarity_search
from components.ask_question import show_ask_question
from components.hybrid_search import show_hybrid_search
from components.risk_analysis import show_risk_analysis
from components.climate_report import show_climate_report
from components.train_model import show_train_model
from components.predict_risks import show_predict_risks
from components.hotspots import show_hotspots
from components.model_status import show_model_status
from components.spatial_events import show_spatial_events
from components.risk_distribution import show_risk_distribution


PAGES = {
    "Home": {
        "🏠 Home": show_home_page,
    },
    "Model": {
        "🧠 Model Status": show_model_status,
        "🎯 Train Model": show_train_model,
    },
    "Graph Creation": {
        "📊 Generate Graph": show_generate_graph,
        "🔍 Graph Schema": show_graph_schema,
        "🗄️ Create Index": show_create_index,
        "📚 Populate Vector Store": show_populate_vectorstore,
    },
    "Search": {
        "🔎 Similarity Search": show_similarity_search,
        "🔀 Hybrid Search": show_hybrid_search,
    },
    "Risk & Climate": {
        "📈 Risk Analysis": show_risk_analysis,
        "❓ Ask Question": show_ask_question,
        "🌡️ Climate Report": show_climate_report,
        "📊 Risk Distribution": show_risk_distribution,
        "🌍 Hotspots": show_hotspots,
        "🗺️ Spatial Events": show_spatial_events,
        "🔮 Predict Risks": show_predict_risks,
    },
}

def main():
    setup_page_config()

    st.sidebar.title("Navigation")
    if "prev_base_url" not in st.session_state:
        st.session_state.prev_base_url = "http://localhost:8000"

    base_url = st.sidebar.text_input(
        "API Base URL", value=st.session_state.prev_base_url, key="base_url"
    )
    if base_url != st.session_state.prev_base_url:
        check_connection(base_url)
        st.session_state.prev_base_url = base_url

    category = st.sidebar.radio("Category", list(PAGES.keys()), key="category")

    
    page = st.sidebar.radio("Page", list(PAGES[category].keys()), key="page")

    
    if category == "Home":
        PAGES[category][page]()
    else:
        PAGES[category][page](base_url)


if __name__ == "__main__":
    main()
