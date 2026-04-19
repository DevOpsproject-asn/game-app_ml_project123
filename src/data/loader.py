"""Data loading utilities for player data"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples=1000, seed=42):
    """
    Generate synthetic player data for development and testing.
    
    Args:
        n_samples (int): Number of player records to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: Synthetic player data
    """
    np.random.seed(seed)
    
    data = {
        'user_id': np.arange(1, n_samples + 1),
        'session_time': np.random.randint(5, 300, n_samples),  # minutes
        'levels_completed': np.random.randint(0, 100, n_samples),
        'in_game_purchases': np.random.exponential(2, n_samples).astype(int),
        'last_login_days_ago': np.random.randint(1, 365, n_samples),
        'total_sessions': np.random.randint(1, 500, n_samples),
        'daily_active': np.random.binomial(1, 0.6, n_samples),
        'player_level': np.random.randint(1, 50, n_samples),
        'total_playtime_hours': np.random.exponential(10, n_samples),
        'achievement_count': np.random.randint(0, 50, n_samples),
        'is_premium': np.random.binomial(1, 0.3, n_samples),
        'churn': np.random.binomial(1, 0.2, n_samples),  # 20% churn rate
        'bot_score': np.random.uniform(0, 1, n_samples),  # 0 = human, 1 = bot
        'avg_session_length': np.random.gamma(2, 15, n_samples).astype(int),
        'purchase_frequency': np.random.poisson(3, n_samples),
    }
    
    df = pd.DataFrame(data)
    logger.info(f"Generated synthetic data with {n_samples} samples")
    return df


def load_player_data(filepath, source_type='csv'):
    """
    Load player data from file.
    
    Args:
        filepath (str): Path to data file
        source_type (str): Type of file ('csv', 'json', 'parquet')
    
    Returns:
        pd.DataFrame: Loaded player data
    """
    try:
        if source_type == 'csv':
            df = pd.read_csv(filepath)
        elif source_type == 'json':
            df = pd.read_json(filepath)
        elif source_type == 'parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file type: {source_type}")
        
        logger.info(f"Loaded data from {filepath}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.warning(f"File not found: {filepath}. Generating synthetic data instead.")
        return generate_synthetic_data()


def save_data(df, filepath, format_type='csv'):
    """
    Save processed data to file.
    
    Args:
        df (pd.DataFrame): Data to save
        filepath (str): Output file path
        format_type (str): Format to save ('csv', 'parquet', 'json')
    """
    if format_type == 'csv':
        df.to_csv(filepath, index=False)
    elif format_type == 'parquet':
        df.to_parquet(filepath, index=False)
    elif format_type == 'json':
        df.to_json(filepath, orient='records')
    
    logger.info(f"Data saved to {filepath}")
