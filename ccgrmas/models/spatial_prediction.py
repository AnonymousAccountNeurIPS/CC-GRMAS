from networkx import radius
from pydantic import BaseModel
from typing import List, Dict, Optional

class TrainingRequest(BaseModel):
    epochs: int = 100
    learning_rate: float = 0.01

class PredictionRequest(BaseModel):
    events: Optional[List[Dict]] = None
    distance_threshold: float = 100.0

class RegionBounds(BaseModel):
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

class HotspotRequest(BaseModel):
    latitude : float
    longitude : float
    radius : float
    grid_resolution: float = 0.1

class RiskDistributionRequest(BaseModel):
    distance_threshold: float