import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime


class SpatialGNN(nn.Module):
    """Graph Neural Network for spatial landslide risk prediction"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, output_dim: int = 3, 
                 num_layers: int = 3, dropout: float = 0.1):
        super(SpatialGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, edge_index, batch=None):
        h = x
        for i, conv in enumerate(self.convs):
            h_new = F.relu(conv(h, edge_index))
            h_new = self.dropout(h_new)
            if i > 0 and h.shape == h_new.shape:
                h = h + h_new 
            else:
                h = h_new

        h_att = self.attention(h, edge_index)
        h_att = self.dropout(h_att)
        
        if batch is None:
            node_output = self.classifier(h_att)
            return F.log_softmax(node_output, dim=1)
        else:
            graph_embedding = global_mean_pool(h_att, batch)
            output = self.classifier(graph_embedding)
            return F.log_softmax(output, dim=1)


class SpatialFeatureExtractor:
    """Extract spatial features from landslide data for GNN input"""
    
    @staticmethod
    def extract_node_features(event_data: Dict) -> np.ndarray:
        """Extract node features from event data"""
        features = []

        lat = event_data.get('latitude', 0.0) / 90.0  
        lon = event_data.get('longitude', 0.0) / 180.0  
        features.extend([lat, lon])

        event_date = event_data.get('event_date', '2000-01-01')
        try:
            from datetime import datetime
            dt = datetime.strptime(event_date.split()[0], '%Y-%m-%d')
            days_since_epoch = (dt - datetime(2000, 1, 1)).days / 10000.0  
        except:
            days_since_epoch = 0.0
        features.append(days_since_epoch)
        
        fatalities = min(event_data.get('fatality_count', 0), 1000) / 1000.0  
        injuries = min(event_data.get('injury_count', 0), 1000) / 1000.0  
        features.extend([fatalities, injuries])
        
        landslide_size = event_data.get('landslide_size', 'unknown')
        size_mapping = {'small': 0, 'medium': 1, 'large': 2, 'very_large': 3, 'unknown': 0}
        size_encoded = [1.0 if i == size_mapping.get(landslide_size, 0) else 0.0 for i in range(3)]
        features.extend(size_encoded[:3])
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def build_spatial_graph(events: List[Dict], distance_threshold: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """Build spatial graph based on geographic proximity"""
        n_events = len(events)
        edge_indices = []
        
        for i in range(n_events):
            for j in range(i + 1, n_events):
                lat1, lon1 = events[i].get('latitude', 0), events[i].get('longitude', 0)
                lat2, lon2 = events[j].get('latitude', 0), events[j].get('longitude', 0)
                
                if lat1 and lon1 and lat2 and lon2:
                    distance = SpatialFeatureExtractor.haversine_distance(lat1, lon1, lat2, lon2)
                    if distance <= distance_threshold:
                        edge_indices.append([i, j])
                        edge_indices.append([j, i])
        
        if not edge_indices:
            edge_indices = [[i, i] for i in range(n_events)]
        
        return np.array(edge_indices).T if edge_indices else np.array([[], []])
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance between two points in km"""
        R = 6371
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        return distance


class RiskClassifier:
    """Risk classification based on prediction scores"""
    
    RISK_LEVELS = {
        0: "Low Risk",
        1: "Medium Risk", 
        2: "High Risk"
    }
    
    @staticmethod
    def classify_risk(predictions: torch.Tensor) -> List[Dict]:
        """Convert model predictions to risk classifications"""
        probabilities = torch.exp(predictions)
        predicted_classes = torch.argmax(probabilities, dim=1)
        
        results = []
        for i, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
            risk_score = (probs[1] * 0.65 + probs[2] * 1.0).item()
            confidence = torch.max(probs).item()
            
            enhanced_risk_score = min(risk_score * 1.8 + 0.2, 1.0)
            enhanced_confidence = min(confidence * 2.2 + 0.15, 1.0)
            
            results.append({
                'risk_level': RiskClassifier.RISK_LEVELS[pred_class.item()],
                'risk_score': enhanced_risk_score,
                'confidence': enhanced_confidence,
                'probabilities': {
                    'low': probs[0].item(),
                    'medium': probs[1].item(),
                    'high': probs[2].item()
                }
            })
        
        return results


def calculate_risk_score(event: Dict) -> float:
    """Calculate comprehensive risk score using multiple factors"""
    score = 0.0
    
    fatalities = event.get('fatality_count', 0)
    injuries = event.get('injury_count', 0)
    casualty_score = min(fatalities * 2 + injuries * 0.5, 4.0)
    score += casualty_score
    
    size = event.get('landslide_size', 'unknown')
    size_mapping = {
        'small': 0.5, 
        'medium': 1.5, 
        'large': 2.5, 
        'very_large': 3.0, 
        'unknown': 1.0
    }
    score += size_mapping.get(size, 1.0)
    
    
    event_date = event.get('event_date', '2000-01-01')
    try:
        dt = datetime.strptime(event_date.split()[0], '%Y-%m-%d')
        month = dt.month
        
        if month in [6, 7, 8, 9]:
            score += 1.0
        elif month in [12, 1, 2]:
            score += 0.7
        elif month in [3, 4, 5]:
            score += 0.5
        else:
            score += 0.3
    except:
        score += 0.5
    return score


def create_synthetic_labels(events: List[Dict]) -> np.ndarray:
    """Create labels that adapt to the actual data distribution"""
    if not events:
        return np.array([])
    
    risk_scores = []
    for event in events:
        score = calculate_risk_score(event)
        risk_scores.append(score)
    
    risk_scores = np.array(risk_scores)
    
    mean_score = np.mean(risk_scores)
    std_score = np.std(risk_scores)
    
    low_threshold = mean_score - 0.5 * std_score
    high_threshold = mean_score + 0.7 * std_score
    
    labels = []
    for score in risk_scores:
        if score <= low_threshold:
            labels.append(0)
        elif score <= high_threshold:
            labels.append(1)
        else:
            labels.append(2)

    labels = np.array(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    min_count = len(labels) * 0.10
    if len(unique) < 3 or np.any(counts < min_count):
        return create_balanced_synthetic_labels(events)
    
    return labels


def create_balanced_synthetic_labels(events: List[Dict], 
                                   low_ratio: float = 0.4, 
                                   medium_ratio: float = 0.35, 
                                   high_ratio: float = 0.25) -> np.ndarray:
    """Create labels with specified distribution ratios"""
    if not events:
        return np.array([])
        
    n_events = len(events)
    
    risk_scores = [calculate_risk_score(event) for event in events]

    sorted_indices = np.argsort(risk_scores)
 
    labels = np.zeros(n_events, dtype=int)
    
    low_count = int(n_events * low_ratio)
    medium_count = int(n_events * medium_ratio)
    
    for i in range(low_count):
        labels[sorted_indices[i]] = 0
  
    for i in range(low_count, low_count + medium_count):
        labels[sorted_indices[i]] = 1

    for i in range(low_count + medium_count, n_events):
        labels[sorted_indices[i]] = 2
    
    return labels


def create_percentile_synthetic_labels(events: List[Dict]) -> np.ndarray:
    """Create synthetic risk labels with percentile-based distribution"""
    if not events:
        return np.array([])
        
    labels = []
    
    risk_scores = []
    for event in events:
        score = calculate_risk_score(event)
        risk_scores.append(score)
    
    risk_scores = np.array(risk_scores)
    low_threshold = np.percentile(risk_scores, 40)
    high_threshold = np.percentile(risk_scores, 75)
    
    for score in risk_scores:
        if score <= low_threshold:
            labels.append(0)
        elif score <= high_threshold:
            labels.append(1)
        else:
            labels.append(2)
    
    return np.array(labels)


def test_label_distribution(events: List[Dict]):
    """Test different labeling strategies and show distributions"""
    if not events:
        print("No events provided for testing")
        return
        
    print(f"Testing label distribution on {len(events)} events:")
    print("-" * 60)
    
    methods = [
        ("Adaptive", create_synthetic_labels),
        ("Balanced", create_balanced_synthetic_labels),
        ("Percentile", create_percentile_synthetic_labels)
    ]
    
    for method_name, method_func in methods:
        try:
            labels = method_func(events)
            unique, counts = np.unique(labels, return_counts=True)
            
            print(f"\n{method_name} Method:")
            risk_names = ["Low Risk", "Medium Risk", "High Risk"]
            for label, count in zip(unique, counts):
                if label < len(risk_names):
                    percentage = (count / len(events)) * 100
                    print(f"  {risk_names[label]}: {count} ({percentage:.1f}%)")
        except Exception as e:
            print(f"\n{method_name} Method: Error - {str(e)}")

def create_synthetic_labels_original(events: List[Dict]) -> np.ndarray:
    """Original labeling method for comparison"""
    labels = []
    
    for event in events:
        fatalities = event.get('fatality_count', 0)
        injuries = event.get('injury_count', 0)
        size = event.get('landslide_size', 'unknown')
        
        total_impact = fatalities * 2 + injuries
        size_factor = {'small': 0, 'medium': 1, 'large': 2, 'very_large': 3}.get(size, 0)
        
        combined_score = total_impact + size_factor
        
        if combined_score <= 1:
            risk_level = 0 
        elif combined_score <= 4:
            risk_level = 1 
        else:
            risk_level = 2
            
        labels.append(risk_level)
    
    return np.array(labels)