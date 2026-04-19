"""Unit tests for ML models"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from src.models import ChurnPredictor, FraudDetector, RecommendationEngine


@pytest.fixture
def synthetic_data():
    """Create synthetic classification data."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        random_state=42
    )
    return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)]), y


def test_churn_predictor_training(synthetic_data):
    """Test churn predictor training."""
    X, y = synthetic_data
    
    model = ChurnPredictor(model_type='xgboost', n_estimators=10, max_depth=3)
    metrics = model.train(X, y, test_size=0.3, cv_folds=3)
    
    assert 'accuracy' in metrics
    assert 'f1' in metrics
    assert 'roc_auc' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1'] <= 1


def test_churn_predictor_prediction(synthetic_data):
    """Test churn predictor prediction."""
    X, y = synthetic_data
    
    model = ChurnPredictor(model_type='random_forest', n_estimators=5)
    model.train(X, y, test_size=0.3)
    
    predictions = model.predict(X[:10])
    probabilities = model.predict(X[:10], return_probability=True)
    
    assert len(predictions) == 10
    assert len(probabilities) == 10
    assert all(p in [0, 1] for p in predictions)
    assert all(0 <= p <= 1 for p in probabilities)


def test_fraud_detector_training(synthetic_data):
    """Test fraud detector training."""
    X, _ = synthetic_data
    
    detector = FraudDetector(method='isolation_forest', contamination=0.1)
    metrics = detector.train(X)
    
    assert 'fraud_rate' in metrics
    assert 'n_fraud_detected' in metrics
    assert 0 <= metrics['fraud_rate'] <= 1


def test_fraud_detector_prediction(synthetic_data):
    """Test fraud detector prediction."""
    X, _ = synthetic_data
    
    detector = FraudDetector(method='isolation_forest')
    detector.train(X)
    
    predictions = detector.predict(X[:20])
    
    assert len(predictions) == 20
    assert all(p in [0, 1] for p in predictions)


def test_recommendation_engine(synthetic_data):
    """Test recommendation engine."""
    X, _ = synthetic_data
    df = pd.DataFrame(X)
    
    engine = RecommendationEngine(method='content_based', n_recommendations=5)
    engine.create_item_features(n_items=20, n_features=5)
    
    recommendations = engine.recommend(
        user_id=1,
        user_data=df,
        liked_items=[0, 1, 2]
    )
    
    assert recommendations['user_id'] == 1
    assert len(recommendations['recommended_items']) <= 5
    assert recommendations['method'] == 'content_based'


def test_model_save_load(synthetic_data, tmp_path):
    """Test model save and load."""
    X, y = synthetic_data
    
    model = ChurnPredictor(model_type='random_forest')
    model.train(X, y)
    
    model_path = tmp_path / "test_model.pkl"
    model.save(str(model_path))
    
    assert model_path.exists()
    
    new_model = ChurnPredictor()
    new_model.load(str(model_path))
    
    predictions1 = model.predict(X[:10])
    predictions2 = new_model.predict(X[:10])
    
    assert np.array_equal(predictions1, predictions2)
