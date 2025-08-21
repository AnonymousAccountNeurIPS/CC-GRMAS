from fastapi import APIRouter, HTTPException
from ccgrmas.models.analytics import (
    RiskAnalysisRequest, RiskAnalysisResponse,
    ClimateReportRequest, ClimateReportResponse
)
from ccgrmas.services.analytics import LangChainAnalyticsService

router = APIRouter()

analytics_service = LangChainAnalyticsService()

@router.post("/risk_analysis", response_model=RiskAnalysisResponse)
async def analyze_risk_patterns(request: RiskAnalysisRequest):
    """Analyze landslide risk patterns with AI insights"""
    try:
        result = analytics_service.analyze_risk_patterns(
            country=request.country,
            trigger=request.trigger
        )
        
        if result["success"]:
            return RiskAnalysisResponse(
                success=True,
                analysis=result["analysis"],
                data_summary=result["data_summary"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/climate_report", response_model=ClimateReportResponse)
async def generate_climate_report(request: ClimateReportRequest):
    """Generate comprehensive climate change impact report"""
    try:
        result = analytics_service.generate_climate_impact_report(region=request.region)
        
        if result["success"]:
            return ClimateReportResponse(
                success=True,
                report=result["report"],
                region=result["region"],
                trigger_analysis=result["trigger_analysis"]
            )
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")