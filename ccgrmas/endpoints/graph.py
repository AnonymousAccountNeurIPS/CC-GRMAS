from fastapi import APIRouter, HTTPException
from ccgrmas.models.graph import GraphRequest, GraphResponse
from ccgrmas.services.graph_service import create_graph
from ccgrmas.constants.grmas import graph

router = APIRouter()

@router.post("/generate_graph", response_model=GraphResponse)
async def generate_graph_endpoint(request: GraphRequest):
    try:
        result = create_graph(
            csv_path=request.csv_path,
            clear_existing=request.clear_existing
        )
        
        if result["success"]:
            return GraphResponse(
                success=True,
                message=result["message"],
                records_processed=result["records_processed"],
                records_failed=result["records_failed"],
                total_records=result["total_records"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@router.get("/graph_schema")
async def graph_schema_endpoint():
    return {"schema": f"{graph.schema}"}