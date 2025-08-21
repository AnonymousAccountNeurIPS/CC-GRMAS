from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class LangChainIndexResponse(BaseModel):
    """Response model for LangChain index creation"""
    success: bool
    message: str
    vector_index: Optional[str] = None
    documents_processed: Optional[int] = None

class VectorStoreResponse(BaseModel):
    """Response model for vector store population"""
    success: bool
    message: str
    processed_count: int
    failed_count: Optional[int] = 0

class LangChainSearchRequest(BaseModel):
    """Request model for LangChain similarity search"""
    query: str
    top_k: Optional[int] = Field(5, description="Number of results to return")
    similarity_threshold: Optional[float] = Field(0.7, description="Similarity threshold for filtering")

class LangChainSearchResponse(BaseModel):
    """Response model for LangChain search results"""
    success: bool
    query: str
    results: List[Dict[str, Any]]
    count: int
    message: Optional[str] = None

class LangChainQARequest(BaseModel):
    """Request model for LangChain QA generation"""
    query: str
    top_k: Optional[int] = Field(5, description="Number of context documents to retrieve")

class LangChainQAResponse(BaseModel):
    """Response model for LangChain QA answers"""
    success: bool
    query: str
    answer: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    source_count: Optional[int] = None
    message: Optional[str] = None

class HybridSearchRequest(BaseModel):
    """Request model for hybrid search combining vector and Cypher"""
    query: str
    cypher_filter: Optional[str] = None
    vector_top_k: Optional[int] = Field(3, description="Number of vector search results")

class HybridSearchResponse(BaseModel):
    """Response model for hybrid search results"""
    success: bool
    query: str
    vector_results: List[Dict[str, Any]]
    cypher_results: List[Dict[str, Any]]
    total_results: int
    message: Optional[str] = None