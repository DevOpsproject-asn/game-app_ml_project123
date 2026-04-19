"""Unit tests for data loading"""
import pytest
import pandas as pd
from src.data.loader import generate_synthetic_data, save_data, load_player_data
import tempfile
import os


def test_generate_synthetic_data():
    """Test synthetic data generation."""
    df = generate_synthetic_data(n_samples=100)
    
    assert len(df) == 100
    assert 'user_id' in df.columns
    assert 'churn' in df.columns
    assert df['user_id'].min() >= 1
    assert df['user_id'].max() <= 100


def test_generate_synthetic_data_columns():
    """Test that generated data has expected columns."""
    df = generate_synthetic_data(n_samples=50)
    
    expected_cols = [
        'user_id', 'session_time', 'levels_completed', 'in_game_purchases',
        'last_login_days_ago', 'total_sessions', 'player_level',
        'total_playtime_hours', 'achievement_count', 'churn'
    ]
    
    for col in expected_cols:
        assert col in df.columns


def test_save_and_load_data():
    """Test saving and loading data."""
    df = generate_synthetic_data(n_samples=100)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, 'test_data.csv')
        
        # Save data
        save_data(df, filepath, format_type='csv')
        assert os.path.exists(filepath)
        
        # Load data
        loaded_df = load_player_data(filepath, source_type='csv')
        
        pd.testing.assert_frame_equal(df, loaded_df)


def test_load_nonexistent_file():
    """Test loading non-existent file returns synthetic data."""
    df = load_player_data('nonexistent_file.csv')
    
    assert len(df) > 0
    assert 'user_id' in df.columns
    assert 'churn' in df.columns
