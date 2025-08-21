from fastapi import APIRouter, HTTPException
from ccgrmas.models.rag import (
    LangChainIndexResponse,
    VectorStoreResponse,
    LangChainSearchRequest, LangChainSearchResponse,
    LangChainQARequest, LangChainQAResponse,
    HybridSearchRequest, HybridSearchResponse
)
from ccgrmas.services.langchain_graphrag_service import LangChainGraphRAGService

router = APIRouter()

langchain_service = LangChainGraphRAGService()

@router.post("/create_index", response_model=LangChainIndexResponse)
async def create_vector_index():
    """Create vector index for LangChain Neo4j integration"""
    try:
        result = langchain_service.create_vector_index()
        
        if result["success"]:
            return LangChainIndexResponse(
                success=True,
                message=result["message"],
                vector_index=result.get("vector_index")
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/populate_vectorstore", response_model=VectorStoreResponse)
async def populate_vector_store():
    """Populate vector store with landslide event documents"""
    try:
        result = langchain_service.populate_vector_store()
        
        if result["success"]:
            return VectorStoreResponse(
                success=True,
                message=result["message"],
                processed_count=result["processed_count"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/similarity_search", response_model=LangChainSearchResponse)
async def similarity_search(request: LangChainSearchRequest):
    """Perform similarity search using LangChain vector store"""
    try:
        result = langchain_service.similarity_search(
            query=request.query, 
            top_k=request.top_k
        )
        
        if result["success"]:
            return LangChainSearchResponse(
                success=True,
                query=result["query"],
                results=result["results"],
                count=result["count"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/ask_question", response_model=LangChainQAResponse)
async def ask_question(request: LangChainQARequest):
    """Ask a question and get AI-generated answer using LangChain QA chain"""
    try:
        result = langchain_service.generate_answer(
            query=request.query, 
            top_k=request.top_k
        )
        
        if result["success"]:
            return LangChainQAResponse(
                success=True,
                query=result["query"],
                answer=result["answer"],
                sources=result["sources"],
                source_count=result["source_count"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/hybrid_search", response_model=HybridSearchResponse)
async def hybrid_search(request: HybridSearchRequest):
    """Perform hybrid search combining vector similarity and Cypher queries"""
    try:
        result = langchain_service.hybrid_search_with_cypher(
            query=request.query,
            cypher_filter=request.cypher_filter
        )
        
        if result["success"]:
            return HybridSearchResponse(
                success=True,
                query=result["query"],
                vector_results=result["vector_results"],
                cypher_results=result["cypher_results"],
                total_results=result["total_results"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")