"""Churn prediction model"""
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)
import xgboost as xgb
import logging

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Predict player churn."""
    
    def __init__(self, model_type='xgboost', **kwargs):
        """
        Initialize churn predictor.
        
        Args:
            model_type (str): 'xgboost', 'random_forest', 'logistic_regression'
            **kwargs: Model hyperparameters
        """
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self.feature_names = None
        
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 7),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
    
    def train(self, X, y, test_size=0.2, cv_folds=5):
        """
        Train churn prediction model.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            y (pd.Series or np.ndarray): Target (0/1)
            test_size (float): Test set proportion
            cv_folds (int): Number of cross-validation folds
        
        Returns:
            dict: Training metrics
        """
        logger.info(f"Training {self.model_type} churn predictor...")
        
        # Store feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='f1')
        self.metrics['cv_f1_scores'] = cv_scores.tolist()
        self.metrics['cv_f1_mean'] = float(cv_scores.mean())
        
        logger.info(f"Churn model training complete. ROC-AUC: {self.metrics['roc_auc']:.4f}")
        
        return self.metrics
    
    def predict(self, X, return_probability=False):
        """
        Predict churn for new data.
        
        Args:
            X (pd.DataFrame or np.ndarray): Features
            return_probability (bool): Return probabilities
        
        Returns:
            np.ndarray: Predictions (0/1) or probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        if return_probability:
            return self.model.predict_proba(X)[:, 1]
        else:
            return self.model.predict(X)
    
    def get_feature_importance(self, top_n=10):
        """Get feature importance."""
        if self.model is None or self.feature_names is None:
            return {}
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importances))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n])
        
        return {}
    
    def save(self, filepath):
        """Save model to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
