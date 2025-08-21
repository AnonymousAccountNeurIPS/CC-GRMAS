from matplotlib.pyplot import grid
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from torch_geometric.data import Data
from ccgrmas.utils.spatial_prediction import SpatialGNN, SpatialFeatureExtractor, RiskClassifier, create_percentile_synthetic_labels
from ccgrmas.constants.grmas import driver
import os


class SpatialPredictionAgent:
    """Main agent for spatial landslide risk prediction using Graph Neural Networks"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.feature_extractor = SpatialFeatureExtractor()
        self.risk_classifier = RiskClassifier()
        self.model_path = model_path or "models/spatial_gnn_model.pth"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def initialize_model(self, input_dim: int = 8, hidden_dim: int = 64, output_dim: int = 3):
        """Initialize the GNN model"""
        self.model = SpatialGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(self.device)
        
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if self.model is None:
                    self.initialize_model()
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                return True
        except Exception as e:
            print(f"Failed to load model: {e}")
        return False
        
    def save_model(self):
        """Save trained model to disk"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'input_dim': self.model.input_dim,
                    'hidden_dim': self.model.hidden_dim,
                    'output_dim': self.model.output_dim
                }
            }, self.model_path)
    
    def fetch_graph_data(self) -> List[Dict]:
        """Fetch landslide event data from Neo4j graph database"""
        query = """
        MATCH (e:Event)
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
               p.landslide_trigger as landslide_trigger,
               g.country_name as country_name,
               g.admin_division_name as admin_division_name
        """
        
        events = []
        with driver.session() as session:
            result = session.run(query)
            for record in result:
                event_data = {}
                for key in record.keys():
                    value = record[key]
                    if value is not None:
                        event_data[key] = value
                events.append(event_data)
        
        return events
    
    def prepare_graph_data(self, events: List[Dict], distance_threshold: float = 100.0) -> Data:
        """Prepare graph data for GNN training/prediction"""
        if not events:
            raise ValueError("No events provided")

        node_features = []
        for event in events:
            features = self.feature_extractor.extract_node_features(event)
            node_features.append(features)
        
        edge_index = self.feature_extractor.build_spatial_graph(events, distance_threshold)

        x = torch.tensor(np.array(node_features), dtype=torch.float).to(self.device)
        edge_index = torch.tensor(edge_index, dtype=torch.long).to(self.device)
        
        return Data(x=x, edge_index=edge_index)
    
    def train_model(self, events: List[Dict], epochs: int = 100, learning_rate: float = 0.01) -> Dict[str, Any]:
        """Train the GNN model on historical landslide data"""
        if not events:
            return {"error": "No training data provided"}
        
        if self.model is None:
            self.initialize_model()
        
        graph_data = self.prepare_graph_data(events)
        labels = torch.tensor(create_percentile_synthetic_labels(events), dtype=torch.long).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = torch.nn.NLLLoss()
        
        self.model.train()
        training_losses = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()

            output = self.model(graph_data.x, graph_data.edge_index)
            loss = criterion(output, labels)

            loss.backward()
            optimizer.step()
            
            training_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        self.save_model()
        
        return {
            "success": True,
            "message": f"Model trained for {epochs} epochs",
            "final_loss": training_losses[-1],
            "training_samples": len(events)
        }
    
    def predict_risk(self, events: List[Dict], distance_threshold: float = 100.0) -> Dict[str, Any]:
        """Predict landslide risk for given events"""
        if not events:
            return {"error": "No events provided for prediction"}
        
        if self.model is None:
            if not self.load_model():
                return {"error": "No trained model available. Please train the model first."}
        
        graph_data = self.prepare_graph_data(events, distance_threshold)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(graph_data.x, graph_data.edge_index)
            risk_classifications = self.risk_classifier.classify_risk(predictions)

        results = []
        for i, (event, classification) in enumerate(zip(events, risk_classifications)):
            results.append({
                "event_id": i,
                "event_title": event.get("event_title", f"Event {i}"),
                "location": {
                    "latitude": event.get("latitude"),
                    "longitude": event.get("longitude"),
                    "country": event.get("country_name"),
                    "region": event.get("admin_division_name")
                },
                "prediction": classification
            })
        
        return {
            "success": True,
            "predictions": results,
            "model_info": {
                "device": str(self.device),
                "total_events": len(events),
                "distance_threshold_km": distance_threshold
            }
        }
    
    def predict_spatial_risk_hotspots(self, region_bounds: Dict[str, float], grid_resolution: float = 0.1) -> Dict[str, Any]:
        """Predict risk hotspots in a spatial region using grid-based sampling"""
        lat_min, lat_max = region_bounds.get('lat_min', 0), region_bounds.get('lat_max', 1)
        lon_min, lon_max = region_bounds.get('lon_min', 0), region_bounds.get('lon_max', 1)
        
        
        grid_points = []
        lat_range = np.arange(lat_min, lat_max + grid_resolution, grid_resolution)
        lon_range = np.arange(lon_min, lon_max + grid_resolution, grid_resolution)
        
        for lat in lat_range:
            for lon in lon_range:
                grid_points.append({
                    'event_title': f'Grid_{lat:.2f}_{lon:.2f}',
                    'latitude': lat,
                    'longitude': lon,
                    'fatality_count': 0,
                    'injury_count': 0,
                    'landslide_size': 'medium'
                })

        if len(grid_points) > 2000:
            grid_points = grid_points[:2000]
        
        predictions = self.predict_risk(grid_points)
        
        if predictions.get("success"):
            hotspots = []
            risk_summary = {"Low Risk": 0, "Medium Risk": 0, "High Risk": 0}
            
            for pred in predictions["predictions"]:
                risk_level = pred["prediction"]["risk_level"]
                risk_score = pred["prediction"]["risk_score"]
                confidence = pred["prediction"]["confidence"]
                
                risk_summary[risk_level] += 1
                
                if risk_score >= 0.15 and confidence > 0.25:
                    hotspots.append({
                        "latitude": pred["location"]["latitude"],
                        "longitude": pred["location"]["longitude"],
                        "risk_level": risk_level,
                        "risk_score": risk_score,
                        "confidence": confidence,
                        "probabilities": pred["prediction"]["probabilities"]
                    })
            
            if len(hotspots) == 0:
                print("No hotspots found with standard criteria, including top confidence predictions")
                all_predictions = sorted(predictions["predictions"], 
                                    key=lambda x: x["prediction"]["risk_score"], reverse=True)
                top_count = max(1, len(all_predictions) // 10)
                
                for pred in all_predictions[:top_count]:
                    hotspots.append({
                        "latitude": pred["location"]["latitude"],
                        "longitude": pred["location"]["longitude"],
                        "risk_level": pred["prediction"]["risk_level"],
                        "risk_score": pred["prediction"]["risk_score"],
                        "confidence": pred["prediction"]["confidence"],
                        "probabilities": pred["prediction"]["probabilities"]
                    })
            
            return {
                "success": True,
                "hotspots": hotspots,
                "region_bounds": region_bounds,
                "grid_resolution": grid_resolution,
                "total_grid_points": len(grid_points),
                "high_risk_points": len([h for h in hotspots if h["risk_level"] == "High Risk"]),
                "medium_risk_points": len([h for h in hotspots if h["risk_level"] == "Medium Risk"]),
                "risk_summary": risk_summary,
                "analysis": {
                    "coverage_area_km2": ((lat_max - lat_min) * 111) * ((lon_max - lon_min) * 111 * np.cos(np.radians((lat_max + lat_min) / 2))),
                    "average_confidence": np.mean([h["confidence"] for h in hotspots]) if hotspots else 0,
                    "average_risk_score": np.mean([h["risk_score"] for h in hotspots]) if hotspots else 0,
                    "hotspot_density": len(hotspots) / len(grid_points) if grid_points else 0
                }
            }
        
        return {"error": "Failed to generate spatial predictions", "details": predictions.get("error", "Unknown error")}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and information"""
        return {
            "model_loaded": self.model is not None,
            "model_path": self.model_path,
            "device": str(self.device),
            "model_file_exists": os.path.exists(self.model_path)
        }
