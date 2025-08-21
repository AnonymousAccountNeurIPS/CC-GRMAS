import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_hybrid_search(base_url: str):
    st.header("Hybrid Search")
    
    query = st.text_input(
        "Search Query:",
        placeholder="Enter your search query...",
        value="What landslides occurred in Jammu and Kashmir?",
        key="hybrid_search_query"
    )
    
    
    if st.button("Search", key="hybrid_search_button"):
        if query:
            with st.spinner("Performing hybrid search..."):
                payload = {"query": query}
                response = make_api_request(
                    base_url,
                    Config.ENDPOINTS["hybrid_search2"],
                    "POST",
                    payload
                )
                data = display_response(response, "Hybrid search completed!")
                
                if data:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Vector Results")
                        for i, result in enumerate(data.get("vector_results", []), 1):
                            with st.expander(f"Vector Result {i}"):
                                st.write(result.get("content", ""))
                                st.json(result.get("metadata", {}))
                    
                    with col2:
                        st.write("### Graph Results")
                        for i, result in enumerate(data.get("cypher_results", []), 1):
                            with st.expander(f"Graph Result {i}"):
                                st.json(result)
        else:
            st.error("Please enter a search query")