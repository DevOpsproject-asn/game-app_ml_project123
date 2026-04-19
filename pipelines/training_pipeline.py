"""Training pipeline for ML models"""
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.loader import generate_synthetic_data, save_data
from src.data.preprocessor import DataPreprocessor
from src.features import FeatureEngineer
from src.models import ChurnPredictor, FraudDetector, RecommendationEngine
from src.services import MLflowService
from src.utils import setup_logging, load_config

logger = logging.getLogger(__name__)


def run_training_pipeline(config_path: str = "configs/config.yaml"):
    """
    Run complete training pipeline for all models.
    
    Args:
        config_path (str): Path to configuration file
    """
    # Setup
    setup_logging(level=logging.INFO)
    config = load_config(config_path)
    
    logger.info("=" * 50)
    logger.info("STARTING ML TRAINING PIPELINE")
    logger.info("=" * 50)
    
    # 1. Load data
    logger.info("\n1. Loading player data...")
    df = generate_synthetic_data(n_samples=2000)
    logger.info(f"Data loaded: {df.shape}")
    
    # 2. Preprocessing
    logger.info("\n2. Preprocessing data...")
    preprocessor = DataPreprocessor(
        scaling_method=config['features']['scaling_method'],
        missing_strategy=config['features']['handle_missing']
    )
    
    feature_columns = ['session_time', 'levels_completed', 'in_game_purchases',
                       'last_login_days_ago', 'total_sessions', 'total_playtime_hours',
                       'player_level', 'achievement_count']
    
    df_processed = preprocessor.preprocess(
        df[feature_columns + ['churn', 'user_id']],
        feature_columns=feature_columns,
        fit=True
    )
    
    # 3. Feature Engineering
    logger.info("\n3. Engineering features...")
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.engineer_features(df_processed[feature_columns + ['churn']])
    
    # 4. Train Churn Model
    logger.info("\n4. Training churn prediction model...")
    churn_features = [col for col in df_features.columns if col != 'churn']
    X_churn = df_features[churn_features]
    y_churn = df_features['churn']
    
    churn_model = ChurnPredictor(
        model_type=config['models']['churn_model']['type'],
        **config['models']['churn_model']['hyperparameters']
    )
    churn_metrics = churn_model.train(X_churn, y_churn)
    
    logger.info(f"Churn Model Metrics:")
    logger.info(f"  - Accuracy: {churn_metrics['accuracy']:.4f}")
    logger.info(f"  - Precision: {churn_metrics['precision']:.4f}")
    logger.info(f"  - Recall: {churn_metrics['recall']:.4f}")
    logger.info(f"  - F1-Score: {churn_metrics['f1']:.4f}")
    logger.info(f"  - ROC-AUC: {churn_metrics['roc_auc']:.4f}")
    
    # Save model
    churn_model.save("models/churn_model.pkl")
    
    # 5. Train Fraud Detection Model
    logger.info("\n5. Training fraud detection model...")
    fraud_features = [col for col in churn_features if col != 'is_premium']
    X_fraud = df_features[fraud_features].fillna(0)
    
    fraud_model = FraudDetector(
        method=config['models']['fraud_model']['type'],
        contamination=config['models']['fraud_model']['contamination'],
        random_state=config['models']['fraud_model']['random_state']
    )
    fraud_metrics = fraud_model.train(X_fraud)
    
    logger.info(f"Fraud Detection Metrics:")
    logger.info(f"  - Fraud Rate: {fraud_metrics['fraud_rate']:.2%}")
    logger.info(f"  - Anomalies Detected: {fraud_metrics['n_fraud_detected']}")
    
    # 6. Setup Recommendation Engine
    logger.info("\n6. Setting up recommendation system...")
    rec_engine = RecommendationEngine(
        method=config['models']['recommendation_model']['type'],
        n_recommendations=config['models']['recommendation_model']['n_recommendations']
    )
    rec_engine.create_user_item_matrix(df)
    rec_engine.create_item_features()
    
    logger.info(f"Recommendation System Ready")
    logger.info(f"  - Method: {rec_engine.method}")
    logger.info(f"  - Recommendations per user: {rec_engine.n_recommendations}")
    
    # 7. Log to MLflow
    logger.info("\n7. Logging experiments to MLflow...")
    try:
        mlflow_service = MLflowService(
            tracking_uri=config['mlflow']['tracking_uri'],
            experiment_name=config['mlflow']['experiment_name']
        )
        
        mlflow_service.log_run(
            run_name="churn_xgboost",
            params=config['models']['churn_model']['hyperparameters'],
            metrics=churn_metrics
        )
        
        mlflow_service.log_run(
            run_name="fraud_detection",
            params={'contamination': fraud_metrics['contamination_setting']},
            metrics={'fraud_rate': fraud_metrics['fraud_rate']}
        )
        
        logger.info("Experiments logged to MLflow")
    except Exception as e:
        logger.warning(f"MLflow logging failed (expected if MLflow not running): {e}")
    
    # 8. Save processed data
    logger.info("\n8. Saving processed data...")
    save_data(df_features, f"{config['data']['processed_data_path']}processed_features.csv")
    
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 50)
    
    return {
        'churn_model': churn_model,
        'fraud_model': fraud_model,
        'recommendation_engine': rec_engine,
        'churn_metrics': churn_metrics,
        'fraud_metrics': fraud_metrics,
        'data': df_features
    }


if __name__ == "__main__":
    results = run_training_pipeline()
