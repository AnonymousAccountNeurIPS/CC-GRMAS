import re
from fastapi import APIRouter, HTTPException, Query
from ccgrmas.models.spatial_prediction import RegionBounds, RiskDistributionRequest, TrainingRequest, PredictionRequest, HotspotRequest
from ccgrmas.services.spatial_prediction_service import SpatialPredictionAgent
from ccgrmas.constants.grmas import driver

router = APIRouter()

spatial_agent = SpatialPredictionAgent()

@router.post("/train")
async def train_spatial_model(request: TrainingRequest):
    """Train the spatial prediction GNN model on historical landslide data"""
    try:
        events = spatial_agent.fetch_graph_data()
        
        if not events:
            raise HTTPException(status_code=400, detail="No training data found in database")
        result = spatial_agent.train_model(
            events=events,
            epochs=request.epochs,
            learning_rate=request.learning_rate
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict")
async def predict_landslide_risk(request: PredictionRequest):
    """Predict landslide risk for specified events or fetch from database"""
    try:
        events = request.events

        if not events:
            events = spatial_agent.fetch_graph_data()
        
        if not events:
            raise HTTPException(status_code=400, detail="No events provided for prediction")

        result = spatial_agent.predict_risk(
            events=events,
            distance_threshold=request.distance_threshold
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/hotspots")
async def predict_spatial_hotspots(request: HotspotRequest):
    """Predict landslide risk hotspots in a geographic region"""
    try:
        
        region_bounds = {
            'lat_min': request.latitude -  request.radius,
            'lat_max': request.latitude +  request.radius,
            'lon_min': request.longitude - request.radius,
            'lon_max': request.longitude + request.radius
        }
        
        result = spatial_agent.predict_spatial_risk_hotspots(
            region_bounds=region_bounds,
            grid_resolution=request.grid_resolution
        )
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Hotspot prediction failed"))
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hotspot prediction failed: {str(e)}")


@router.get("/model/status")
async def get_model_status():
    """Get current status of the spatial prediction model"""
    try:
        status = spatial_agent.get_model_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model status: {str(e)}")


@router.post("/events/spatial")
async def get_spatial_events(request: RegionBounds):
    """Get landslide events within a spatial bounding box"""
    try:
        query = """
        MATCH (e:Event)
        WHERE e.latitude >= $lat_min AND e.latitude <= $lat_max
        AND e.longitude >= $lon_min AND e.longitude <= $lon_max
        OPTIONAL MATCH (e)-[:LOCATED_NEAR]->(g:GazetteerPoint)
        OPTIONAL MATCH (e)-[:HAS_PROFILE]->(p:LandslideProfile)
        RETURN e.event_title as event_title,
               e.latitude as latitude,
               e.longitude as longitude,
               e.event_date as event_date,
               e.fatality_count as fatality_count,
               e.injury_count as injury_count,
               p.landslide_size as landslide_size,
               p.landslide_category as landslide_category,
               g.country_name as country_name,
               g.admin_division_name as admin_division_name
        """
        
        events = []
        with driver.session() as session:
            result = session.run(query, 
                               lat_min=request.lat_min, lat_max=request.lat_max,
                               lon_min=request.lon_min, lon_max=request.lon_max)

            for record in result:
                event_data = {}
                for key in record.keys():
                    value = record[key]
                    if value is not None:
                        event_data[key] = value
                events.append(event_data)
        
        return {
            "success": True,
            "events": events,
            "bounds": {
                "lat_min": request.lat_min,
                "lat_max": request.lat_max,
                "lon_min": request.lon_min,
                "lon_max": request.lon_max
            },
            "count": len(events)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch spatial events: {str(e)}")


@router.post("/analytics/risk-distribution")
async def get_risk_distribution(request : RiskDistributionRequest):
    """Get risk distribution analytics from recent predictions"""
    try:
        events = spatial_agent.fetch_graph_data()
        
        if not events:
            return {"message": "No data available for analysis"}

        result = spatial_agent.predict_risk(events, request.distance_threshold)
        
        if not result.get("success"):
            raise HTTPException(status_code=400, detail="Failed to generate risk analysis")

        risk_counts = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
        country_risks = {}
        
        for prediction in result["predictions"]:
            risk_level = prediction["prediction"]["risk_level"]
            risk_counts[risk_level] += 1
            
            country = prediction["location"].get("country", "Unknown")
            if country not in country_risks:
                country_risks[country] = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
            country_risks[country][risk_level] += 1
        
        return {
            "success": True,
            "risk_distribution": risk_counts,
            "country_risk_breakdown": country_risks,
            "total_analyzed": len(result["predictions"]),
            "high_risk_percentage": (risk_counts["High Risk"] / len(result["predictions"])) * 100
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Risk analysis failed: {str(e)}")