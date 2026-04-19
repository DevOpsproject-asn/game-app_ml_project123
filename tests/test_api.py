"""Unit tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_root_endpoint(client):
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_model_health(client):
    """Test model health endpoint."""
    response = client.get("/api/v1/health/models")
    assert response.status_code == 200
    data = response.json()
    assert "churn_model" in data
    assert "recommendation_model" in data
    assert "fraud_detector" in data


def test_predict_churn(client):
    """Test churn prediction endpoint."""
    payload = {
        "player_data": {
            "user_id": 1,
            "session_time": 45,
            "levels_completed": 25,
            "in_game_purchases": 5,
            "last_login_days_ago": 10,
            "total_sessions": 50,
            "daily_active": 1,
            "player_level": 15,
            "total_playtime_hours": 25.5,
            "achievement_count": 10,
            "is_premium": 1
        }
    }
    
    response = client.post("/api/v1/predict-churn", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == 1
    assert 0 <= data["churn_probability"] <= 1
    assert data["risk_level"] in ["low", "medium", "high"]
    assert "recommendation" in data


def test_detect_fraud(client):
    """Test fraud detection endpoint."""
    payload = {
        "user_id": 1,
        "session_time": 45,
        "levels_completed": 25,
        "in_game_purchases": 5,
        "total_playtime_hours": 25.5,
        "player_level": 15
    }
    
    response = client.post("/api/v1/detect-fraud", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == 1
    assert 0 <= data["fraud_score"] <= 1
    assert data["is_fraud"] in [True, False]
    assert data["risk_level"] in ["low", "medium", "high"]
    assert isinstance(data["flags"], list)


def test_get_recommendations(client):
    """Test recommendation endpoint."""
    payload = {
        "user_id": 1,
        "liked_items": [1, 5, 10],
        "n_recommendations": 5
    }
    
    response = client.post("/api/v1/recommend", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == 1
    assert isinstance(data["recommended_items"], list)
    assert len(data["recommended_items"]) <= 5
    assert data["method"] in ["content_based", "collaborative"]
    assert data["confidence"] in ["high", "low"]


def test_player_insights(client):
    """Test player insights endpoint."""
    payload = {
        "user_id": 1,
        "total_sessions": 50,
        "levels_completed": 25,
        "in_game_purchases": 5,
        "last_login_days_ago": 10,
        "total_playtime_hours": 25.5,
        "achievement_count": 10
    }
    
    response = client.post("/api/v1/player-insights", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["user_id"] == 1
    assert data["engagement_level"] in ["low", "medium", "high"]
    assert isinstance(data["lifetime_value_estimate"], (int, float))
    assert data["retention_risk"] in ["low", "medium", "high"]
    assert isinstance(data["recommended_actions"], list)


def test_invalid_payload(client):
    """Test endpoint with invalid payload."""
    payload = {"invalid": "data"}
    response = client.post("/api/v1/predict-churn", json=payload)
    assert response.status_code == 422  # Validation error
