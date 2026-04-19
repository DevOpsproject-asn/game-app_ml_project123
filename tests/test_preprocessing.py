"""Unit tests for preprocessing"""
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor


@pytest.fixture
def sample_data():
    """Create sample data with missing values."""
    return pd.DataFrame({
        'feature1': [1, 2, np.nan, 4, 5],
        'feature2': [10, np.nan, 30, 40, 50],
        'feature3': [100, 200, 300, 400, 500]
    })


def test_handle_missing_values_mean(sample_data):
    """Test handling missing values with mean strategy."""
    preprocessor = DataPreprocessor(missing_strategy='mean')
    result = preprocessor.handle_missing_values(sample_data.copy())
    
    assert result.isnull().sum().sum() == 0
    assert result.loc[2, 'feature1'] == sample_data['feature1'].mean()


def test_handle_missing_values_drop(sample_data):
    """Test handling missing values with drop strategy."""
    preprocessor = DataPreprocessor(missing_strategy='drop')
    result = preprocessor.handle_missing_values(sample_data.copy())
    
    assert result.isnull().sum().sum() == 0
    assert len(result) < len(sample_data)


def test_remove_outliers(sample_data):
    """Test outlier removal."""
    data_with_outliers = sample_data.copy()
    data_with_outliers.loc[0, 'feature1'] = 1000  # Outlier
    
    preprocessor = DataPreprocessor()
    result = preprocessor.remove_outliers(
        data_with_outliers,
        columns=['feature1'],
        threshold=1.5
    )
    
    assert len(result) <= len(data_with_outliers)


def test_scale_features_standardscaler(sample_data):
    """Test feature scaling with StandardScaler."""
    preprocessor = DataPreprocessor(scaling_method='standardscaler')
    result = preprocessor.scale_features(
        sample_data.copy(),
        feature_columns=['feature1', 'feature2', 'feature3'],
        fit=True
    )
    
    # Check that mean is close to 0 and std close to 1
    assert abs(result[['feature1', 'feature2', 'feature3']].mean().mean()) < 0.1


def test_complete_preprocessing_pipeline(sample_data):
    """Test complete preprocessing pipeline."""
    preprocessor = DataPreprocessor(
        scaling_method='standardscaler',
        missing_strategy='mean'
    )
    
    result = preprocessor.preprocess(
        sample_data.copy(),
        feature_columns=['feature1', 'feature2', 'feature3']
    )
    
    assert result.isnull().sum().sum() == 0
    assert len(result.columns) == 3
    assert len(result) > 0
