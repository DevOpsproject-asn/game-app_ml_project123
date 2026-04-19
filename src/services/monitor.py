"""Model monitoring and drift detection"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelMonitor:
    """Monitor model performance and detect drift."""
    
    def __init__(self, threshold=0.1):
        """Initialize monitor."""
        self.threshold = threshold
        self.baseline_stats = None
        self.predictions_log = []
    
    def set_baseline(self, X, model_predictions):
        """Set baseline statistics."""
        self.baseline_stats = {
            'prediction_mean': model_predictions.mean(),
            'prediction_std': model_predictions.std(),
            'feature_means': X.mean(numeric_only=True).to_dict() if isinstance(X, pd.DataFrame) else {},
            'timestamp': datetime.now()
        }
        logger.info("Baseline statistics set")
    
    def detect_drift(self, X, model_predictions):
        """Detect data drift."""
        if self.baseline_stats is None:
            logger.warning("No baseline set for drift detection")
            return False
        
        # Check prediction distribution drift
        pred_mean_diff = abs(model_predictions.mean() - self.baseline_stats['prediction_mean'])
        drift_detected = pred_mean_diff > self.threshold
        
        if drift_detected:
            logger.warning(f"Data drift detected! Mean difference: {pred_mean_diff:.4f}")
        
        return drift_detected
    
    def log_prediction(self, user_id, prediction, probability, model_name):
        """Log prediction."""
        log_entry = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'prediction': prediction,
            'probability': probability,
            'model_name': model_name
        }
        self.predictions_log.append(log_entry)
    
    def get_model_stats(self):
        """Get model statistics."""
        if not self.predictions_log:
            return {}
        
        df = pd.DataFrame(self.predictions_log)
        return {
            'total_predictions': len(df),
            'avg_probability': df['probability'].mean(),
            'positive_rate': (df['prediction'] == 1).mean(),
            'models_used': df['model_name'].unique().tolist()
        }
