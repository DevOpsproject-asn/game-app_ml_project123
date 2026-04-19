"""Feature engineering module"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature engineering for player data."""
    
    def __init__(self):
        self.feature_names = []
    
    def create_engagement_features(self, df):
        """Create engagement-based features."""
        df['session_per_day'] = df['total_sessions'] / (df['last_login_days_ago'] + 1)
        df['avg_level_per_session'] = df['levels_completed'] / (df['total_sessions'] + 1)
        df['playtime_per_session'] = df['total_playtime_hours'] / (df['total_sessions'] + 1)
        df['purchase_intensity'] = df['in_game_purchases'] / (df['total_sessions'] + 1)
        
        return df
    
    def create_temporal_features(self, df):
        """Create time-based features."""
        df['days_inactive'] = df['last_login_days_ago']
        df['is_recently_active'] = (df['last_login_days_ago'] <= 7).astype(int)
        df['is_weekly_active'] = (df['last_login_days_ago'] <= 30).astype(int)
        df['is_monthly_active'] = (df['last_login_days_ago'] <= 90).astype(int)
        
        return df
    
    def create_progression_features(self, df):
        """Create game progression features."""
        df['progression_ratio'] = df['levels_completed'] / (df['player_level'] + 1)
        df['achievement_ratio'] = df['achievement_count'] / (df['levels_completed'] + 1)
        df['premium_multiplier'] = 1 + df['is_premium']  # Premium players worth more
        
        return df
    
    def create_monetization_features(self, df):
        """Create monetization-related features."""
        df['lifetime_value_score'] = (
            df['in_game_purchases'] * 10 + 
            df['total_playtime_hours'] * 2 + 
            df['is_premium'] * 100
        )
        df['purchase_ratio'] = df['in_game_purchases'] / (df['total_sessions'] + 1)
        df['high_value_player'] = (df['lifetime_value_score'] > df['lifetime_value_score'].median()).astype(int)
        
        return df
    
    def create_behavioral_features(self, df):
        """Create behavior-based features."""
        df['consistency_score'] = 1 / (1 + np.exp(-((df['total_sessions'] - 50) / 20)))  # Sigmoid
        df['engagement_level'] = pd.cut(
            df['total_playtime_hours'], 
            bins=3, 
            labels=['low', 'medium', 'high'],
            ordered=True
        )
        df['engagement_level'] = df['engagement_level'].cat.codes
        
        return df
    
    def create_risk_features(self, df):
        """Create churn risk features."""
        df['inactivity_risk'] = df['last_login_days_ago'] / 365
        df['engagement_risk'] = 1 - (df['total_sessions'] / (df['total_sessions'].max() + 1))
        df['churn_risk_score'] = 0.6 * df['inactivity_risk'] + 0.4 * df['engagement_risk']
        
        return df
    
    def create_anomaly_features(self, df):
        """Create features for anomaly/fraud detection."""
        df['session_time_zscore'] = (df['avg_session_length'] - df['avg_session_length'].mean()) / (df['avg_session_length'].std() + 1)
        df['purchase_zscore'] = (df['in_game_purchases'] - df['in_game_purchases'].mean()) / (df['in_game_purchases'].std() + 1)
        df['level_progression_speed'] = df['levels_completed'] / (df['total_playtime_hours'] + 1)
        df['unusual_activity'] = ((np.abs(df['session_time_zscore']) > 2) | 
                                  (np.abs(df['purchase_zscore']) > 2)).astype(int)
        
        return df
    
    def engineer_features(self, df, drop_original=False):
        """
        Apply all feature engineering transformations.
        
        Args:
            df (pd.DataFrame): Input data
            drop_original (bool): Whether to drop original features
        
        Returns:
            pd.DataFrame: Data with engineered features
        """
        logger.info("Starting feature engineering...")
        
        df = self.create_engagement_features(df)
        df = self.create_temporal_features(df)
        df = self.create_progression_features(df)
        df = self.create_monetization_features(df)
        df = self.create_behavioral_features(df)
        df = self.create_risk_features(df)
        df = self.create_anomaly_features(df)
        
        logger.info(f"Feature engineering complete. Total features: {df.shape[1]}")
        
        return df
    
    def get_feature_importance_info(self, feature_names):
        """Get information about engineered features."""
        categories = {
            'engagement': ['session_per_day', 'avg_level_per_session', 'playtime_per_session', 'purchase_intensity'],
            'temporal': ['days_inactive', 'is_recently_active', 'is_weekly_active', 'is_monthly_active'],
            'progression': ['progression_ratio', 'achievement_ratio', 'premium_multiplier'],
            'monetization': ['lifetime_value_score', 'purchase_ratio', 'high_value_player'],
            'behavioral': ['consistency_score', 'engagement_level'],
            'risk': ['inactivity_risk', 'engagement_risk', 'churn_risk_score'],
            'anomaly': ['session_time_zscore', 'purchase_zscore', 'level_progression_speed', 'unusual_activity']
        }
        
        return categories
