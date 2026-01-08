"""
Machine Learning-Based Protocol Selection for Federated Learning

This module implements supervised ML models to predict the optimal protocol
based on historical training data and performance metrics.

Approach:
1. Collect training data: (network conditions, resources, mobility) -> best protocol
2. Train classifier: Random Forest, XGBoost, Neural Network
3. Predict optimal protocol for new conditions
4. Continuously retrain with new data
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


class MLProtocolSelector:
    """
    Machine Learning-based protocol selector using supervised learning.
    
    Features used for prediction:
    - Network: latency, bandwidth, packet_loss, jitter
    - Resources: cpu_usage, memory_usage, battery_level
    - Model: model_size_mb
    - Mobility: velocity, connection_stability
    - Context: time_of_day, network_type
    
    Target: protocol_name (mqtt, amqp, grpc, quic, dds)
    Performance metric: convergence_time, final_accuracy, total_cost
    """
    
    def __init__(self, model_type: str = "random_forest", model_path: str = None):
        """
        Initialize ML-based selector.
        
        Args:
            model_type: Type of ML model ('random_forest', 'gradient_boosting', 'neural_network')
            model_path: Path to saved model (if loading existing model)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'latency', 'bandwidth', 'packet_loss', 'jitter',
            'cpu_usage', 'memory_usage', 'battery_level', 'is_charging',
            'model_size_mb', 'update_frequency',
            'velocity', 'connection_stability', 'handoffs_per_hour',
            'hour_of_day', 'is_wifi', 'is_cellular'
        ]
        self.protocol_names = ['mqtt', 'amqp', 'grpc', 'quic', 'dds']
        self.training_history = []
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced classes
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == "neural_network":
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def extract_features(self, conditions: Dict) -> np.ndarray:
        """
        Extract feature vector from condition dictionary.
        
        Args:
            conditions: Dictionary with network, resources, model, mobility info
            
        Returns:
            Feature vector (numpy array)
        """
        network = conditions.get('network', {})
        resources = conditions.get('resources', {})
        model = conditions.get('model', {})
        mobility = conditions.get('mobility', {})
        context = conditions.get('context', {})
        
        features = [
            # Network features
            network.get('latency', 50),
            network.get('bandwidth', 10),
            network.get('packet_loss', 0),
            network.get('jitter', 0),
            
            # Resource features
            resources.get('cpu_usage', 50),
            resources.get('memory_usage', 50),
            resources.get('battery_level', 100),
            int(resources.get('is_charging', True)),
            
            # Model features
            model.get('model_size_mb', 10),
            model.get('update_frequency', 60),
            
            # Mobility features
            mobility.get('velocity', 0),
            mobility.get('connection_stability', 100),
            mobility.get('handoffs_per_hour', 0),
            
            # Context features
            context.get('hour_of_day', datetime.now().hour),
            int(context.get('is_wifi', True)),
            int(context.get('is_cellular', False))
        ]
        
        return np.array(features).reshape(1, -1)
    
    def predict_protocol(
        self, 
        conditions: Dict,
        return_probabilities: bool = False
    ) -> Tuple[str, float]:
        """
        Predict the best protocol for given conditions.
        
        Args:
            conditions: Current system conditions
            return_probabilities: Whether to return class probabilities
            
        Returns:
            (protocol_name, confidence_score) or (protocol_name, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Train model first or load pretrained model.")
        
        # Extract and scale features
        features = self.extract_features(conditions)
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        protocol = self.protocol_names[prediction]
        
        # Get confidence/probabilities
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = probabilities[prediction]
            
            if return_probabilities:
                prob_dict = {p: prob for p, prob in zip(self.protocol_names, probabilities)}
                return protocol, prob_dict
            else:
                return protocol, confidence
        else:
            return protocol, 1.0
    
    def train(
        self,
        training_data: List[Dict],
        validation_split: float = 0.2,
        verbose: bool = True
    ) -> Dict:
        """
        Train the ML model on collected data.
        
        Args:
            training_data: List of training examples, each with:
                - conditions: Dict with network/resource/mobility info
                - protocol: The protocol that was used
                - performance: Dict with convergence_time, accuracy, etc.
            validation_split: Fraction of data for validation
            verbose: Print training progress
            
        Returns:
            Training results dictionary
        """
        if len(training_data) < 10:
            raise ValueError(f"Need at least 10 training examples, got {len(training_data)}")
        
        # Prepare training data
        X = []
        y = []
        weights = []  # Performance-based sample weights
        
        for example in training_data:
            features = self.extract_features(example['conditions'])
            X.append(features.flatten())
            
            protocol = example['protocol']
            y.append(self.protocol_names.index(protocol))
            
            # Weight samples by performance (better performance = higher weight)
            performance = example.get('performance', {})
            # Normalize: faster convergence + higher accuracy = better
            conv_time = performance.get('convergence_time', 100)
            accuracy = performance.get('final_accuracy', 0.5)
            
            # Weight: inversely proportional to time, proportional to accuracy
            weight = accuracy / (conv_time + 1)  # Add 1 to avoid division by zero
            weights.append(weight)
        
        X = np.array(X)
        y = np.array(y)
        weights = np.array(weights)
        
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        # Split data
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, weights, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Fit scaler on training data
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        if verbose:
            print(f"\n{'='*70}")
            print(f"Training {self.model_type.upper()} Model")
            print(f"{'='*70}")
            print(f"Training samples: {len(X_train)}")
            print(f"Validation samples: {len(X_val)}")
            print(f"Features: {len(self.feature_names)}")
            print(f"Classes: {len(self.protocol_names)}")
        
        self.model.fit(X_train_scaled, y_train, sample_weight=w_train)
        
        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        val_pred = self.model.predict(X_val_scaled)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        
        results = {
            'train_accuracy': train_acc,
            'validation_accuracy': val_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        if verbose:
            print(f"\nTraining Results:")
            print(f"  Training Accuracy: {train_acc:.4f}")
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            print(f"\nValidation Classification Report:")
            print(classification_report(
                y_val, val_pred, 
                target_names=self.protocol_names,
                zero_division=0
            ))
            
            # Feature importance (if available)
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]  # Top 10 features
                
                print(f"\nTop 10 Most Important Features:")
                for i, idx in enumerate(indices, 1):
                    print(f"  {i}. {self.feature_names[idx]}: {importances[idx]:.4f}")
            
            print(f"{'='*70}\n")
        
        self.training_history.append(results)
        return results
    
    def incremental_update(
        self,
        new_data: List[Dict],
        learning_rate: float = 0.1,
        verbose: bool = True
    ):
        """
        Incrementally update the model with new data (online learning).
        
        Args:
            new_data: New training examples
            learning_rate: How much to weight new data vs existing model
            verbose: Print update info
        """
        if self.model is None:
            raise ValueError("Model must be trained first before incremental updates")
        
        # Extract features from new data
        X_new = []
        y_new = []
        
        for example in new_data:
            features = self.extract_features(example['conditions'])
            X_new.append(features.flatten())
            
            protocol = example['protocol']
            y_new.append(self.protocol_names.index(protocol))
        
        X_new = np.array(X_new)
        y_new = np.array(y_new)
        
        # Scale features
        X_new_scaled = self.scaler.transform(X_new)
        
        # Partial fit (if supported)
        if hasattr(self.model, 'partial_fit'):
            self.model.partial_fit(X_new_scaled, y_new)
            if verbose:
                print(f"✓ Model updated with {len(new_data)} new examples (partial fit)")
        else:
            # For models that don't support partial_fit, retrain with combined data
            # This is a simplified approach - in production, maintain a replay buffer
            if verbose:
                print(f"⚠ Model type '{self.model_type}' doesn't support partial_fit")
                print(f"  Recommendation: Periodically retrain with full dataset")
    
    def save_model(self, path: str):
        """Save trained model and scaler to disk"""
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        save_dict = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'protocol_names': self.protocol_names,
            'training_history': self.training_history
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_dict, path)
        print(f"✓ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from disk"""
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        save_dict = joblib.load(path)
        
        self.model = save_dict['model']
        self.scaler = save_dict['scaler']
        self.model_type = save_dict['model_type']
        self.feature_names = save_dict['feature_names']
        self.protocol_names = save_dict['protocol_names']
        self.training_history = save_dict.get('training_history', [])
        
        print(f"✓ Model loaded from {path}")
        print(f"  Type: {self.model_type}")
        print(f"  Training history: {len(self.training_history)} sessions")


# ==================== Data Collection Helper ====================

class ProtocolPerformanceCollector:
    """
    Collects training data by running experiments with different protocols
    and recording performance.
    """
    
    def __init__(self, data_file: str = "protocol_training_data.json"):
        self.data_file = Path(data_file)
        self.data = self.load_data()
    
    def load_data(self) -> List[Dict]:
        """Load existing training data"""
        if self.data_file.exists():
            with open(self.data_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_data(self):
        """Save training data to disk"""
        self.data_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def record_experiment(
        self,
        conditions: Dict,
        protocol: str,
        performance: Dict
    ):
        """
        Record results from a single FL experiment.
        
        Args:
            conditions: System conditions during experiment
            protocol: Protocol that was used
            performance: Results (convergence_time, final_accuracy, etc.)
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'conditions': conditions,
            'protocol': protocol,
            'performance': performance
        }
        
        self.data.append(entry)
        self.save_data()
        
        print(f"✓ Recorded experiment: {protocol} -> "
              f"time={performance.get('convergence_time', 0):.1f}s, "
              f"acc={performance.get('final_accuracy', 0):.4f}")
    
    def get_training_data(self) -> List[Dict]:
        """Get all collected data for training"""
        return self.data
    
    def get_statistics(self) -> Dict:
        """Get statistics about collected data"""
        if not self.data:
            return {"total_samples": 0}
        
        protocol_counts = {}
        for entry in self.data:
            protocol = entry['protocol']
            protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1
        
        return {
            'total_samples': len(self.data),
            'protocol_distribution': protocol_counts,
            'date_range': {
                'first': self.data[0]['timestamp'],
                'last': self.data[-1]['timestamp']
            }
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    print("="*70)
    print("MACHINE LEARNING-BASED PROTOCOL SELECTION")
    print("="*70)
    
    # Example 1: Generate synthetic training data
    print("\n1. Generating synthetic training data...")
    
    collector = ProtocolPerformanceCollector("example_training_data.json")
    
    # Simulate different scenarios
    scenarios = [
        # IoT device, low bandwidth -> MQTT is best
        {
            'conditions': {
                'network': {'latency': 100, 'bandwidth': 2, 'packet_loss': 1, 'jitter': 10},
                'resources': {'cpu_usage': 80, 'memory_usage': 85, 'battery_level': 30, 'is_charging': False},
                'model': {'model_size_mb': 10, 'update_frequency': 120},
                'mobility': {'velocity': 0, 'connection_stability': 70, 'handoffs_per_hour': 0},
                'context': {'hour_of_day': 14, 'is_wifi': False, 'is_cellular': True}
            },
            'best_protocol': 'mqtt',
            'performance': {'convergence_time': 45.2, 'final_accuracy': 0.85}
        },
        # High-performance server -> gRPC is best
        {
            'conditions': {
                'network': {'latency': 5, 'bandwidth': 100, 'packet_loss': 0, 'jitter': 1},
                'resources': {'cpu_usage': 20, 'memory_usage': 30, 'battery_level': 100, 'is_charging': True},
                'model': {'model_size_mb': 100, 'update_frequency': 60},
                'mobility': {'velocity': 0, 'connection_stability': 100, 'handoffs_per_hour': 0},
                'context': {'hour_of_day': 2, 'is_wifi': True, 'is_cellular': False}
            },
            'best_protocol': 'grpc',
            'performance': {'convergence_time': 25.8, 'final_accuracy': 0.92}
        },
        # Mobile device -> QUIC is best
        {
            'conditions': {
                'network': {'latency': 80, 'bandwidth': 5, 'packet_loss': 2, 'jitter': 20},
                'resources': {'cpu_usage': 50, 'memory_usage': 60, 'battery_level': 45, 'is_charging': False},
                'model': {'model_size_mb': 30, 'update_frequency': 90},
                'mobility': {'velocity': 60, 'connection_stability': 50, 'handoffs_per_hour': 5},
                'context': {'hour_of_day': 18, 'is_wifi': False, 'is_cellular': True}
            },
            'best_protocol': 'quic',
            'performance': {'convergence_time': 38.5, 'final_accuracy': 0.88}
        }
    ]
    
    # Generate 50 samples with variations
    for _ in range(50):
        for scenario in scenarios:
            # Add noise to conditions
            conditions = {
                'network': {k: v + np.random.normal(0, v*0.1) for k, v in scenario['conditions']['network'].items()},
                'resources': {k: v + np.random.normal(0, v*0.1) if isinstance(v, (int, float)) else v 
                             for k, v in scenario['conditions']['resources'].items()},
                'model': {k: v + np.random.normal(0, v*0.05) for k, v in scenario['conditions']['model'].items()},
                'mobility': {k: v + np.random.normal(0, max(1, v*0.1)) for k, v in scenario['conditions']['mobility'].items()},
                'context': scenario['conditions']['context']
            }
            
            # Add noise to performance
            performance = {
                'convergence_time': scenario['performance']['convergence_time'] + np.random.normal(0, 5),
                'final_accuracy': min(1.0, max(0.0, scenario['performance']['final_accuracy'] + np.random.normal(0, 0.02)))
            }
            
            collector.record_experiment(conditions, scenario['best_protocol'], performance)
    
    print(f"\n✓ Generated {len(collector.data)} training samples")
    print(f"  Statistics: {collector.get_statistics()}")
    
    # Example 2: Train ML model
    print("\n2. Training ML model...")
    
    selector = MLProtocolSelector(model_type="random_forest")
    training_results = selector.train(
        collector.get_training_data(),
        validation_split=0.2,
        verbose=True
    )
    
    # Example 3: Make predictions
    print("\n3. Making predictions on new conditions...")
    
    test_conditions = {
        'network': {'latency': 60, 'bandwidth': 3, 'packet_loss': 1.5, 'jitter': 15},
        'resources': {'cpu_usage': 70, 'memory_usage': 75, 'battery_level': 35, 'is_charging': False},
        'model': {'model_size_mb': 20, 'update_frequency': 100},
        'mobility': {'velocity': 20, 'connection_stability': 65, 'handoffs_per_hour': 2},
        'context': {'hour_of_day': 16, 'is_wifi': False, 'is_cellular': True}
    }
    
    protocol, probabilities = selector.predict_protocol(test_conditions, return_probabilities=True)
    
    print(f"\n🎯 Predicted Protocol: {protocol.upper()}")
    print(f"\nProbabilities:")
    for proto, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        bar = '█' * int(prob * 50)
        print(f"  {proto:6s}: {prob:5.2%} {bar}")
    
    # Example 4: Save model
    print("\n4. Saving trained model...")
    selector.save_model("models/protocol_selector_rf.pkl")
    
    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)
