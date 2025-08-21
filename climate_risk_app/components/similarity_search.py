import streamlit as st
from config import Config
from api_utils import make_api_request, display_response

def show_similarity_search(base_url: str):
    st.header("Similarity Search")
    
    query = st.text_input(
        "Search Query:",
        placeholder="Enter keywords or phrases to find similar landslide events...",
        value="What landslides occurred in Jammu and Kashmir?",
        key="similarity_search_query"
    )
    
    top_k = st.number_input("Number of results", min_value=1, max_value=20, value=5, key="similarity_search_top_k")
    
    if st.button("Search", key="similarity_search_button"):
        if query:
            with st.spinner("Searching similar events..."):
                response = make_api_request(
                    base_url,
                    Config.ENDPOINTS["similarity_search"],
                    "POST",
                    {"query": query, "top_k": top_k}
                )
                data = display_response(response, "Search completed!")
                
                if data and data.get("results"):
                    st.write(f"### Found {len(data['results'])} similar events")
                    
                    for i, result in enumerate(data["results"], 1):
                        with st.expander(f"Result {i}"):
                            st.write("**Content:**", result.get("content", ""))
                            if result.get("metadata"):
                                st.json(result["metadata"])
        else:
            st.error("Please enter a search query")