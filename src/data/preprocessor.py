"""Data preprocessing module"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handle data cleaning and preprocessing."""
    
    def __init__(self, scaling_method='standardscaler', missing_strategy='mean'):
        """
        Initialize preprocessor.
        
        Args:
            scaling_method (str): 'standardscaler' or 'minmaxscaler'
            missing_strategy (str): 'mean', 'median', or 'drop'
        """
        self.scaling_method = scaling_method
        self.missing_strategy = missing_strategy
        self.scaler = None
        self.feature_columns = None
        
    def handle_missing_values(self, df):
        """Handle missing values in dataframe."""
        initial_missing = df.isnull().sum().sum()
        
        if self.missing_strategy == 'drop':
            df = df.dropna()
        elif self.missing_strategy == 'mean':
            df = df.fillna(df.mean(numeric_only=True))
        elif self.missing_strategy == 'median':
            df = df.fillna(df.median(numeric_only=True))
        
        final_missing = df.isnull().sum().sum()
        logger.info(f"Missing values: {initial_missing} -> {final_missing}")
        return df
    
    def remove_outliers(self, df, columns, method='iqr', threshold=1.5):
        """Remove outliers using IQR method."""
        if method == 'iqr':
            for col in columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        logger.info(f"Outliers removed. New shape: {df.shape}")
        return df
    
    def scale_features(self, df, feature_columns, fit=True):
        """
        Scale features using specified method.
        
        Args:
            df (pd.DataFrame): Input dataframe
            feature_columns (list): Columns to scale
            fit (bool): Whether to fit scaler or use existing
        
        Returns:
            pd.DataFrame: Dataframe with scaled features
        """
        if self.scaling_method == 'standardscaler':
            scaler = StandardScaler()
        elif self.scaling_method == 'minmaxscaler':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.scaling_method}")
        
        if fit:
            self.scaler = scaler
            self.feature_columns = feature_columns
            df[feature_columns] = scaler.fit_transform(df[feature_columns])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            df[feature_columns] = self.scaler.transform(df[feature_columns])
        
        logger.info(f"Features scaled using {self.scaling_method}")
        return df
    
    def preprocess(self, df, feature_columns, fit=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw data
            feature_columns (list): Numeric columns to process
            fit (bool): Whether to fit scalers
        
        Returns:
            pd.DataFrame: Preprocessed data
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Handle missing values
        df = self.handle_missing_values(df)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        logger.info(f"Duplicates removed: {initial_rows} -> {len(df)} rows")
        
        # Remove outliers (optional, can be configured)
        outlier_columns = [col for col in feature_columns 
                          if col in df.columns and df[col].dtype in ['int64', 'float64']]
        df = self.remove_outliers(df, outlier_columns)
        
        # Scale features
        df = self.scale_features(df, feature_columns, fit=fit)
        
        logger.info("Preprocessing complete!")
        return df
    
    def get_feature_statistics(self, df):
        """Get summary statistics of features."""
        return df.describe()
