"""Exploratory data analysis notebook for player data"""
# This notebook demonstrates:
# 1. Data loading and exploration
# 2. Exploratory data analysis
# 3. Feature engineering
# 4. Model training basics
# 5. Results visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.loader import generate_synthetic_data
from src.data.preprocessor import DataPreprocessor
from src.features import FeatureEngineer
from src.models import ChurnPredictor

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
df = generate_synthetic_data(n_samples=1000)
print(f"Data shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# Check for missing values
print(f"\nMissing values:\n{df.isnull().sum()}")

# Churn distribution
print(f"\nChurn distribution:")
print(df['churn'].value_counts())
print(f"Churn rate: {df['churn'].mean():.2%}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Churn distribution
df['churn'].value_counts().plot(kind='bar', ax=axes[0, 0])
axes[0, 0].set_title('Churn Distribution')
axes[0, 0].set_ylabel('Count')

# Playtime by churn
df.boxplot(column='total_playtime_hours', by='churn', ax=axes[0, 1])
axes[0, 1].set_title('Playtime by Churn Status')

# Purchases vs Churn
df.boxplot(column='in_game_purchases', by='churn', ax=axes[1, 0])
axes[1, 0].set_title('Purchases by Churn Status')

# Levels completed vs Churn
df.boxplot(column='levels_completed', by='churn', ax=axes[1, 1])
axes[1, 1].set_title('Levels Completed by Churn Status')

plt.tight_layout()
plt.show()

# Feature engineering
engineer = FeatureEngineer()
feature_cols = ['session_time', 'levels_completed', 'in_game_purchases',
                'last_login_days_ago', 'total_sessions', 'total_playtime_hours',
                'player_level', 'achievement_count', 'is_premium']
df_engineered = engineer.engineer_features(df.copy())
print(f"\nEngineered features shape: {df_engineered.shape}")
print(f"New features added: {df_engineered.shape[1] - df.shape[1]}")

# Train model
churn_model = ChurnPredictor(model_type='xgboost')
X = df_engineered[[col for col in df_engineered.columns if col != 'churn']]
y = df_engineered['churn']

metrics = churn_model.train(X, y)
print(f"\nModel Performance:")
for metric, value in metrics.items():
    if isinstance(value, (int, float)):
        print(f"  {metric}: {value:.4f}")

# Feature importance
importance = churn_model.get_feature_importance(top_n=10)
print(f"\nTop 10 Important Features:")
for i, (feature, imp) in enumerate(importance.items(), 1):
    print(f"  {i}. {feature}: {imp:.4f}")
