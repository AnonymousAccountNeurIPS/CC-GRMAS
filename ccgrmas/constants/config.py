import os
from typing import List
from dotenv import load_dotenv

load_dotenv()

class LangChainConfig:
    """Configuration class for LangChain GraphRAG settings"""
    
    def __init__(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        self.embedding_model = "models/embedding-001"
        self.llm_model = "gemini-1.5-pro"
        self.temperature = 0.1
        self.max_tokens = 2048
        
        self.vector_index_name = "landslide_vector_index"
        self.fulltext_index_name = "landslide_fulltext_index"
        self.embedding_dimension = 768 
        self.node_label = "Event"
        self.text_property = "text_content"
        self.embedding_property = "embedding_vector"

        self.default_top_k = 5
        self.similarity_threshold = 0.7
        
        self.prompt_template = """
You are an expert climate scientist and geologist specializing in landslide risk analysis. 
Use the following context about landslide events to answer the question comprehensively.

Context from Landslide Database:
{context}

Question: {question}

Please provide a detailed analysis that includes:
1. Direct answers based on the provided landslide data
2. Patterns and trends you observe in the data
3. Risk assessment and climate change implications
4. Geographic and temporal correlations
5. Recommendations for risk management

Focus on:
- Event characteristics (location, date, casualties)
- Landslide profiles (type, trigger, size, setting)
- Geographic distribution and patterns
- Temporal trends and seasonal patterns
- Trigger mechanisms and their relation to climate
- Impact assessment (fatalities, injuries)

If the provided context doesn't contain sufficient information for any aspect, 
clearly state what additional data would be needed.

Answer:
"""