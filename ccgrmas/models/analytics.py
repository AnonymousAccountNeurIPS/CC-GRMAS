from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class RiskAnalysisRequest(BaseModel):
    """Request model for risk pattern analysis"""
    country: Optional[str] = Field(None, description="Filter by country")
    trigger: Optional[str] = Field(None, description="Filter by landslide trigger")

class RiskAnalysisResponse(BaseModel):
    """Response model for risk analysis"""
    success: bool
    analysis: Optional[str] = None
    data_summary: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class ClimateReportRequest(BaseModel):
    """Request model for climate impact report"""
    region: Optional[str] = Field(None, description="Region for focused analysis")

class ClimateReportResponse(BaseModel):
    """Response model for climate impact report"""
    success: bool
    report: Optional[str] = None
    region: str
    trigger_analysis: Optional[List[Dict[str, Any]]] = None
    message: Optional[str] = None