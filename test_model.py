#!/usr/bin/env python3
"""
Spatial GNN Model Evaluation Script
Comprehensive evaluation of the trained spatial GNN model for landslide risk prediction
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings("ignore")

try:
    from ccgrmas.services.spatial_prediction_service import SpatialPredictionAgent
    from ccgrmas.utils.spatial_prediction import (
        SpatialGNN, SpatialFeatureExtractor, RiskClassifier,
        create_percentile_synthetic_labels
    )
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure the ccgrmas modules are in your Python path")
    sys.exit(1)


class SpatialGNNEvaluator:
    """Comprehensive evaluator for Spatial GNN model"""
    
    def __init__(self, model_path: str = "models/spatial_gnn_model.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_extractor = SpatialFeatureExtractor()
        self.risk_classifier = RiskClassifier()
        self.spatial_agent = SpatialPredictionAgent(model_path)
        
        # Risk level mapping
        self.risk_levels = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
        
        print(f"Initializing evaluator with device: {self.device}")
        print(f"Model path: {self.model_path}")
    
    def load_model_and_data(self) -> Tuple[bool, List[Dict]]:
        """Load the trained model and fetch evaluation data"""
        print("Loading model and fetching data...")
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False, []
        
        # Load model
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model_config = checkpoint.get('model_config', {})
            
            self.model = SpatialGNN(
                input_dim=model_config.get('input_dim', 8),
                hidden_dim=model_config.get('hidden_dim', 64),
                output_dim=model_config.get('output_dim', 3)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print("✓ Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False, []
        
        # Fetch data from database
        events = self.spatial_agent.fetch_graph_data()
        
        if not events:
            print("Error: No events found in database")
            return False, []
        
        print(f"✓ Loaded {len(events)} events from database")
        return True, events
    
    def prepare_evaluation_data(self, events: List[Dict], 
                              test_size: float = 0.1, 
                              distance_threshold: float = 100.0) -> Dict[str, Any]:
        """Prepare data for evaluation with train/test split"""
        print(f"Preparing evaluation data with {len(events)} events...")
        
        # Create synthetic labels
        labels = create_percentile_synthetic_labels(events)
        
        # Split data
        train_events, test_events, train_labels, test_labels = train_test_split(
            events, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Prepare graph data for both sets
        train_graph = self.spatial_agent.prepare_graph_data(train_events, distance_threshold)
        test_graph = self.spatial_agent.prepare_graph_data(test_events, distance_threshold)
        
        print(f"✓ Data split - Train: {len(train_events)}, Test: {len(test_events)}")
        print(f"✓ Label distribution - Train: {np.bincount(train_labels)}, Test: {np.bincount(test_labels)}")
        
        return {
            'train_events': train_events,
            'test_events': test_events,
            'train_labels': torch.tensor(train_labels, dtype=torch.long),
            'test_labels': torch.tensor(test_labels, dtype=torch.long),
            'train_graph': train_graph,
            'test_graph': test_graph
        }
    
    def evaluate_model_performance(self, eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model performance on test data"""
        print("Evaluating model performance...")
        
        self.model.eval()
        with torch.no_grad():
            # Get predictions
            train_logits = self.model(eval_data['train_graph'].x, eval_data['train_graph'].edge_index)
            test_logits = self.model(eval_data['test_graph'].x, eval_data['test_graph'].edge_index)
            
            # Convert to probabilities and predictions
            train_probs = torch.exp(train_logits).cpu().numpy()
            test_probs = torch.exp(test_logits).cpu().numpy()
            
            train_preds = torch.argmax(train_logits, dim=1).cpu().numpy()
            test_preds = torch.argmax(test_logits, dim=1).cpu().numpy()
            
            train_labels_np = eval_data['train_labels'].cpu().numpy()
            test_labels_np = eval_data['test_labels'].cpu().numpy()
        
        # Calculate metrics
        train_accuracy = accuracy_score(train_labels_np, train_preds)
        test_accuracy = accuracy_score(test_labels_np, test_preds)
        
        # Detailed metrics for test set
        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
            test_labels_np, test_preds, average='weighted'
        )
        
        # Per-class metrics
        class_report = classification_report(
            test_labels_np, test_preds, 
            target_names=list(self.risk_levels.values()),
            output_dict=True
        )
        
        # Confusion matrix
        conf_matrix = confusion_matrix(test_labels_np, test_preds)
        
        results = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'train_probs': train_probs,
            'test_probs': test_probs,
            'train_preds': train_preds,
            'test_preds': test_preds,
            'train_labels': train_labels_np,
            'test_labels': test_labels_np
        }
        
        print(f"✓ Test Accuracy: {test_accuracy:.4f}")
        print(f"✓ Test Precision: {test_precision:.4f}")
        print(f"✓ Test Recall: {test_recall:.4f}")
        print(f"✓ Test F1-Score: {test_f1:.4f}")
        
        return results
    
    def plot_evaluation_results(self, results: Dict[str, Any], save_dir: str = "evaluation_plots"):
        """Create comprehensive evaluation plots"""
        print("Creating evaluation plots...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', 
                   xticklabels=list(self.risk_levels.values()),
                   yticklabels=list(self.risk_levels.values()),
                   cmap='Blues')
        plt.title('Confusion Matrix - Spatial GNN Model')
        plt.xlabel('Predicted Risk Level')
        plt.ylabel('True Risk Level')
        plt.tight_layout()
        plt.savefig(f"{save_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Metrics Bar Chart
        metrics_data = {
            'Accuracy': results['test_accuracy'],
            'Precision': results['test_precision'],
            'Recall': results['test_recall'],
            'F1-Score': results['test_f1']
        }
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(metrics_data.keys(), metrics_data.values(), 
                      color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Per-Class Performance
        class_metrics = []
        for risk_level in self.risk_levels.values():
            if risk_level in results['classification_report']:
                class_metrics.append({
                    'Risk Level': risk_level,
                    'Precision': results['classification_report'][risk_level]['precision'],
                    'Recall': results['classification_report'][risk_level]['recall'],
                    'F1-Score': results['classification_report'][risk_level]['f1-score']
                })
        
        if class_metrics:
            df_metrics = pd.DataFrame(class_metrics)
            
            plt.figure(figsize=(12, 6))
            x = np.arange(len(df_metrics))
            width = 0.25
            
            plt.bar(x - width, df_metrics['Precision'], width, label='Precision', alpha=0.8)
            plt.bar(x, df_metrics['Recall'], width, label='Recall', alpha=0.8)
            plt.bar(x + width, df_metrics['F1-Score'], width, label='F1-Score', alpha=0.8)
            
            plt.xlabel('Risk Level')
            plt.ylabel('Score')
            plt.title('Per-Class Performance Metrics')
            plt.xticks(x, df_metrics['Risk Level'])
            plt.legend()
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/per_class_metrics.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Prediction Distribution
        plt.figure(figsize=(15, 5))
        
        # True vs Predicted distribution
        plt.subplot(1, 3, 1)
        true_counts = np.bincount(results['test_labels'])
        pred_counts = np.bincount(results['test_preds'])
        
        x = np.arange(len(self.risk_levels))
        width = 0.35
        
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        plt.xlabel('Risk Level')
        plt.ylabel('Count')
        plt.title('True vs Predicted Distribution')
        plt.xticks(x, list(self.risk_levels.values()), rotation=45)
        plt.legend()
        
        # Probability distributions
        plt.subplot(1, 3, 2)
        for i, risk_level in self.risk_levels.items():
            plt.hist(results['test_probs'][:, i], alpha=0.7, label=f'{risk_level}', bins=20)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Probability Distributions')
        plt.legend()
        
        # Prediction confidence
        plt.subplot(1, 3, 3)
        max_probs = np.max(results['test_probs'], axis=1)
        plt.hist(max_probs, bins=20, alpha=0.7, color='purple')
        plt.xlabel('Maximum Probability (Confidence)')
        plt.ylabel('Frequency')
        plt.title('Prediction Confidence Distribution')
        plt.axvline(np.mean(max_probs), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(max_probs):.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/prediction_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Plots saved to {save_dir}/")
    
    def analyze_model_errors(self, results: Dict[str, Any], eval_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model errors and misclassifications"""
        print("Analyzing model errors...")
        
        test_labels = results['test_labels']
        test_preds = results['test_preds']
        test_probs = results['test_probs']
        test_events = eval_data['test_events']
        
        # Find misclassified samples
        misclassified = test_labels != test_preds
        misclassified_indices = np.where(misclassified)[0]
        
        error_analysis = {
            'total_errors': len(misclassified_indices),
            'error_rate': len(misclassified_indices) / len(test_labels),
            'error_breakdown': {}
        }
        
        # Analyze errors by true class
        for true_class in range(len(self.risk_levels)):
            true_mask = test_labels == true_class
            errors_in_class = misclassified & true_mask
            
            if np.sum(true_mask) > 0:
                error_analysis['error_breakdown'][self.risk_levels[true_class]] = {
                    'total_samples': np.sum(true_mask),
                    'errors': np.sum(errors_in_class),
                    'error_rate': np.sum(errors_in_class) / np.sum(true_mask)
                }
        
        # Analyze most confident wrong predictions
        if len(misclassified_indices) > 0:
            max_probs_wrong = np.max(test_probs[misclassified_indices], axis=1)
            most_confident_wrong_idx = misclassified_indices[np.argmax(max_probs_wrong)]
            
            error_analysis['most_confident_error'] = {
                'index': int(most_confident_wrong_idx),
                'true_label': self.risk_levels[test_labels[most_confident_wrong_idx]],
                'predicted_label': self.risk_levels[test_preds[most_confident_wrong_idx]],
                'confidence': float(max_probs_wrong[np.argmax(max_probs_wrong)]),
                'event_info': {
                    'title': test_events[most_confident_wrong_idx].get('event_title', 'Unknown'),
                    'country': test_events[most_confident_wrong_idx].get('country_name', 'Unknown'),
                    'fatalities': test_events[most_confident_wrong_idx].get('fatality_count', 0),
                    'injuries': test_events[most_confident_wrong_idx].get('injury_count', 0)
                }
            }
        
        return error_analysis
    
    def generate_evaluation_report(self, results: Dict[str, Any], 
                                 error_analysis: Dict[str, Any],
                                 eval_data: Dict[str, Any],
                                 save_path: str = "model_evaluation_report.txt") -> str:
        """Generate comprehensive evaluation report"""
        print("Generating evaluation report...")
        
        report = []
        report.append("=" * 80)
        report.append("SPATIAL GNN MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model Info
        report.append("MODEL INFORMATION:")
        report.append(f"  Model Path: {self.model_path}")
        report.append(f"  Device: {self.device}")
        report.append(f"  Model exists: {os.path.exists(self.model_path)}")
        report.append("")
        
        # Dataset Info
        train_size = len(eval_data['train_events'])
        test_size = len(eval_data['test_events'])
        total_size = train_size + test_size
        
        report.append("DATASET INFORMATION:")
        report.append(f"  Total Events: {total_size}")
        report.append(f"  Training Set: {train_size} ({train_size/total_size*100:.1f}%)")
        report.append(f"  Test Set: {test_size} ({test_size/total_size*100:.1f}%)")
        report.append("")
        
        # Overall Performance
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Training Accuracy: {results['train_accuracy']:.4f}")
        report.append(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        report.append(f"  Test Precision: {results['test_precision']:.4f}")
        report.append(f"  Test Recall: {results['test_recall']:.4f}")
        report.append(f"  Test F1-Score: {results['test_f1']:.4f}")
        report.append("")
        
        # Per-Class Performance
        report.append("PER-CLASS PERFORMANCE:")
        for risk_level in self.risk_levels.values():
            if risk_level in results['classification_report']:
                metrics = results['classification_report'][risk_level]
                report.append(f"  {risk_level}:")
                report.append(f"    Precision: {metrics['precision']:.4f}")
                report.append(f"    Recall: {metrics['recall']:.4f}")
                report.append(f"    F1-Score: {metrics['f1-score']:.4f}")
                report.append(f"    Support: {metrics['support']}")
        report.append("")
        
        # Confusion Matrix
        report.append("CONFUSION MATRIX:")
        report.append("  " + "\t".join([f"{level:>12}" for level in self.risk_levels.values()]))
        for i, row in enumerate(results['confusion_matrix']):
            risk_level = self.risk_levels[i]
            row_str = f"{risk_level:>12}\t" + "\t".join([f"{val:>12}" for val in row])
            report.append("  " + row_str)
        report.append("")
        
        # Error Analysis
        report.append("ERROR ANALYSIS:")
        report.append(f"  Total Errors: {error_analysis['total_errors']}")
        report.append(f"  Error Rate: {error_analysis['error_rate']:.4f}")
        report.append("")
        
        report.append("  Error Breakdown by Class:")
        for class_name, error_info in error_analysis['error_breakdown'].items():
            report.append(f"    {class_name}:")
            report.append(f"      Total Samples: {error_info['total_samples']}")
            report.append(f"      Errors: {error_info['errors']}")
            report.append(f"      Error Rate: {error_info['error_rate']:.4f}")
        
        if 'most_confident_error' in error_analysis:
            mce = error_analysis['most_confident_error']
            report.append("")
            report.append("  Most Confident Misclassification:")
            report.append(f"    Event: {mce['event_info']['title']}")
            report.append(f"    Country: {mce['event_info']['country']}")
            report.append(f"    Fatalities: {mce['event_info']['fatalities']}")
            report.append(f"    Injuries: {mce['event_info']['injuries']}")
            report.append(f"    True Label: {mce['true_label']}")
            report.append(f"    Predicted Label: {mce['predicted_label']}")
            report.append(f"    Confidence: {mce['confidence']:.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        # Save report
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Report saved to {save_path}")
        return report_text
    
    def run_full_evaluation(self, distance_threshold: float = 100.0, 
                          test_size: float = 0.1) -> Dict[str, Any]:
        """Run complete evaluation pipeline"""
        print("Starting full model evaluation...")
        print("=" * 60)
        
        # Load model and data
        success, events = self.load_model_and_data()
        if not success:
            return {"error": "Failed to load model or data"}
        
        # Prepare evaluation data
        eval_data = self.prepare_evaluation_data(events, test_size, distance_threshold)
        
        # Evaluate performance
        results = self.evaluate_model_performance(eval_data)
        
        # Analyze errors
        error_analysis = self.analyze_model_errors(results, eval_data)
        
        # Create plots
        self.plot_evaluation_results(results)
        
        # Generate report
        report = self.generate_evaluation_report(results, error_analysis, eval_data)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print("=" * 60)
        
        return {
            'success': True,
            'results': results,
            'error_analysis': error_analysis,
            'report': report,
            'eval_data': eval_data
        }


def main():
    """Main evaluation function"""
    print("Spatial GNN Model Evaluation")
    print("=" * 40)
    
    # Configuration
    model_path = "models/spatial_gnn_model.pth"
    distance_threshold = 100.0  # km
    test_size = 0.1
    
    # Create evaluator
    evaluator = SpatialGNNEvaluator(model_path)
    
    # Run evaluation
    evaluation_results = evaluator.run_full_evaluation(
        distance_threshold=distance_threshold,
        test_size=test_size
    )
    
    if evaluation_results.get('success'):
        print("\nEvaluation completed successfully!")
        print("\nKey Results:")
        results = evaluation_results['results']
        print(f"  Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  Test F1-Score: {results['test_f1']:.4f}")
        print(f"  Error Rate: {evaluation_results['error_analysis']['error_rate']:.4f}")
    else:
        print(f"Evaluation failed: {evaluation_results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()