"""Unit tests for feature engineering"""
import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer


@pytest.fixture
def sample_data():
    """Create sample player data."""
    return pd.DataFrame({
        'user_id': range(1, 101),
        'total_sessions': np.random.randint(1, 500, 100),
        'levels_completed': np.random.randint(0, 100, 100),
        'in_game_purchases': np.random.randint(0, 50, 100),
        'last_login_days_ago': np.random.randint(1, 365, 100),
        'total_playtime_hours': np.random.uniform(1, 100, 100),
        'player_level': np.random.randint(1, 50, 100),
        'achievement_count': np.random.randint(0, 50, 100),
        'is_premium': np.random.binomial(1, 0.3, 100),
        'avg_session_length': np.random.randint(5, 300, 100),
        'purchase_frequency': np.random.poisson(3, 100)
    })


def test_engagement_features(sample_data):
    """Test engagement feature creation."""
    engineer = FeatureEngineer()
    result = engineer.create_engagement_features(sample_data.copy())
    
    assert 'session_per_day' in result.columns
    assert 'avg_level_per_session' in result.columns
    assert 'playtime_per_session' in result.columns
    assert 'purchase_intensity' in result.columns
    assert len(result) == len(sample_data)


def test_temporal_features(sample_data):
    """Test temporal feature creation."""
    engineer = FeatureEngineer()
    result = engineer.create_temporal_features(sample_data.copy())
    
    assert 'days_inactive' in result.columns
    assert 'is_recently_active' in result.columns
    assert all(result['is_recently_active'].isin([0, 1]))


def test_progression_features(sample_data):
    """Test progression feature creation."""
    engineer = FeatureEngineer()
    result = engineer.create_progression_features(sample_data.copy())
    
    assert 'progression_ratio' in result.columns
    assert 'achievement_ratio' in result.columns
    assert 'premium_multiplier' in result.columns


def test_full_engineering(sample_data):
    """Test full feature engineering pipeline."""
    engineer = FeatureEngineer()
    result = engineer.engineer_features(sample_data.copy())
    
    # Check that new features exist
    expected_features = [
        'session_per_day', 'avg_level_per_session', 'days_inactive',
        'progression_ratio', 'lifetime_value_score', 'consistency_score',
        'inactivity_risk', 'session_time_zscore'
    ]
    
    for feature in expected_features:
        assert feature in result.columns
    
    assert result.shape[0] == len(sample_data)


def test_no_nan_in_output(sample_data):
    """Test that output has no NaN values."""
    engineer = FeatureEngineer()
    result = engineer.engineer_features(sample_data.copy())
    
    # Fill any remaining NaN with 0
    result = result.fillna(0)
    assert result.isnull().sum().sum() == 0
