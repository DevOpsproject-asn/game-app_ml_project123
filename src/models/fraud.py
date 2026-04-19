"""Fraud detection and anomaly detection"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class FraudDetector:
    """Detect fraudulent and bot players."""
    
    def __init__(self, method='isolation_forest', contamination=0.05, **kwargs):
        """
        Initialize fraud detector.
        
        Args:
            method (str): 'isolation_forest' or 'one_class_svm'
            contamination (float): Expected fraud rate
            **kwargs: Model hyperparameters
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.metrics = {}
        
        if method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=contamination,
                random_state=kwargs.get('random_state', 42)
            )
        elif method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=contamination,
                kernel='rbf',
                gamma=kwargs.get('gamma', 'auto')
            )
    
    def detect_rule_based_fraud(self, df):
        """
        Apply rule-based fraud detection heuristics.
        
        Args:
            df (pd.DataFrame): Player data
        
        Returns:
            pd.Series: Fraud flags (0/1)
        """
        fraud_indicators = pd.Series(0, index=df.index)
        
        # Rule 1: Impossible session duration (>12 hours non-stop)
        if 'session_time' in df.columns:
            fraud_indicators[df['session_time'] > 720] = 1
        
        # Rule 2: Unrealistic progression (too fast level completion)
        if 'levels_completed' in df.columns and 'total_playtime_hours' in df.columns:
            levels_per_hour = df['levels_completed'] / (df['total_playtime_hours'] + 1)
            fraud_indicators[levels_per_hour > 50] = 1
        
        # Rule 3: Suspicious purchase pattern (too many purchases in short time)
        if 'in_game_purchases' in df.columns and 'total_sessions' in df.columns:
            purchase_per_session = df['in_game_purchases'] / (df['total_sessions'] + 1)
            fraud_indicators[purchase_per_session > 5] = 1
        
        # Rule 4: Bot score threshold (if bot_score exists)
        if 'bot_score' in df.columns:
            fraud_indicators[df['bot_score'] > 0.8] = 1
        
        return fraud_indicators
    
    def train(self, X, y=None):
        """
        Train fraud detection model.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (np.ndarray): Labels (optional, for validation)
        
        Returns:
            dict: Training metrics
        """
        logger.info(f"Training {self.method} fraud detector...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Predictions
        predictions = self.model.predict(X_scaled)
        predictions = (predictions == -1).astype(int)  # Convert to 0/1
        
        fraud_rate = predictions.mean()
        logger.info(f"Detected fraud rate: {fraud_rate:.2%}")
        
        self.metrics = {
            'fraud_rate': float(fraud_rate),
            'n_fraud_detected': int(predictions.sum()),
            'contamination_setting': self.contamination
        }
        
        return self.metrics
    
    def predict(self, X, return_scores=False):
        """
        Predict fraud for new data.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            return_scores (bool): Return anomaly scores
        
        Returns:
            np.ndarray: Fraud predictions (0/1) or scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self.scaler.transform(X)
        
        if return_scores:
            # Return anomaly scores (lower = more anomalous for Isolation Forest)
            if hasattr(self.model, 'score_samples'):
                scores = self.model.score_samples(X_scaled)
                return -scores  # Invert so higher = more anomalous
        
        predictions = self.model.predict(X_scaled)
        return (predictions == -1).astype(int)
    
    def detect_combined(self, df, use_rule_based=True, use_ml=True, rule_weight=0.3, ml_weight=0.7):
        """
        Combined fraud detection (rule-based + ML).
        
        Args:
            df (pd.DataFrame): Player data
            use_rule_based (bool): Use rule-based detection
            use_ml (bool): Use ML-based detection
            rule_weight (float): Weight for rule-based
            ml_weight (float): Weight for ML-based
        
        Returns:
            np.ndarray: Fraud scores (0-1)
        """
        scores = np.zeros(len(df))
        
        if use_rule_based:
            rule_fraud = self.detect_rule_based_fraud(df).values
            scores += rule_weight * rule_fraud
        
        if use_ml:
            # Get only numeric columns for ML model
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            X = df[numeric_cols].fillna(0)
            ml_fraud = self.predict(X)
            scores += ml_weight * ml_fraud
        
        # Normalize to 0-1
        scores = np.clip(scores / (rule_weight + ml_weight), 0, 1)
        
        return scores
    
    def get_fraud_insights(self, df, fraud_scores):
        """Get insights about detected fraud."""
        fraud_mask = fraud_scores > 0.5
        
        insights = {
            'total_players': len(df),
            'flagged_as_fraud': fraud_mask.sum(),
            'fraud_percentage': (fraud_mask.sum() / len(df) * 100),
            'high_risk_count': (fraud_scores > 0.8).sum(),
            'medium_risk_count': ((fraud_scores > 0.5) & (fraud_scores <= 0.8)).sum(),
            'low_risk_count': ((fraud_scores > 0.3) & (fraud_scores <= 0.5)).sum(),
            'avg_fraud_score': fraud_scores.mean()
        }
        
        return insights
