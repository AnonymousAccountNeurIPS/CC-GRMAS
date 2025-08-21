import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_ask_question(base_url: str):
    st.header("AI Question Answering")
    

    query = st.text_input(
        "Search Query:",
        placeholder="Enter keywords or phrases to find similar landslide events...",
        value="Which countries have the highest fatality rates?",
        key="similarity_search_query"
    )
    
    top_k = st.number_input("Sources to use", min_value=1, max_value=10, value=5, key="ask_question_top_k")
    
    if st.button("Get Answer", key="ask_question_button"):
        if query:
            with st.spinner("Analyzing data and generating answer..."):
                response = make_api_request(
                    base_url,
                    Config.ENDPOINTS["ask_question"],
                    "POST",
                    {"query": query, "top_k": top_k}
                )
                data = display_response(response, "Answer generated successfully!")
                
                if data:
                    st.write("### Answer")
                    st.markdown(data.get("answer", "No answer generated"))
                    
                    if data.get("sources"):
                        st.write("### Sources")
                        for i, source in enumerate(data["sources"], 1):
                            with st.expander(f"Source {i}"):
                                st.write("**Content:**", source.get("content", ""))
                                st.json(source.get("metadata", {}))
        else:
            st.error("Please enter a question")