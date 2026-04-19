"""Quick start script for the Game ML Platform"""
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import generate_synthetic_data
from src.data.preprocessor import DataPreprocessor
from src.features import FeatureEngineer
from src.models import ChurnPredictor, FraudDetector, RecommendationEngine
from src.utils import setup_logging

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_demo():
    """Run a quick demo of the ML system."""
    
    logger.info("=" * 60)
    logger.info("GAME ML PLATFORM - QUICK DEMO")
    logger.info("=" * 60)
    
    # 1. Load data
    logger.info("\n[1] Loading synthetic player data...")
    df = generate_synthetic_data(n_samples=500)
    logger.info(f"    ✓ Loaded {len(df)} player records")
    logger.info(f"    ✓ Churn rate: {df['churn'].mean():.1%}")
    
    # 2. Preprocess
    logger.info("\n[2] Preprocessing data...")
    preprocessor = DataPreprocessor(missing_strategy='mean')
    feature_cols = ['session_time', 'levels_completed', 'in_game_purchases',
                   'last_login_days_ago', 'total_sessions', 'total_playtime_hours',
                   'player_level', 'achievement_count']
    df_processed = preprocessor.preprocess(
        df[feature_cols + ['churn', 'user_id']],
        feature_columns=feature_cols,
        fit=True
    )
    logger.info(f"    ✓ Preprocessing complete")
    
    # 3. Feature engineering
    logger.info("\n[3] Engineering features...")
    engineer = FeatureEngineer()
    df_features = engineer.engineer_features(df_processed)
    logger.info(f"    ✓ Created {df_features.shape[1]} total features")
    
    # 4. Train churn model
    logger.info("\n[4] Training churn prediction model...")
    X_churn = df_features[[col for col in df_features.columns if col != 'churn']]
    y_churn = df_features['churn']
    
    churn_model = ChurnPredictor(model_type='xgboost', n_estimators=50)
    metrics = churn_model.train(X_churn, y_churn, test_size=0.2, cv_folds=3)
    
    logger.info(f"    ✓ Model trained!")
    logger.info(f"    ✓ ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"    ✓ F1-Score: {metrics['f1']:.4f}")
    
    # 5. Train fraud detector
    logger.info("\n[5] Training fraud detection model...")
    fraud_model = FraudDetector(method='isolation_forest', contamination=0.05)
    fraud_metrics = fraud_model.train(X_churn)
    
    logger.info(f"    ✓ Fraud detector trained!")
    logger.info(f"    ✓ Fraud rate detected: {fraud_metrics['fraud_rate']:.2%}")
    
    # 6. Setup recommendation engine
    logger.info("\n[6] Setting up recommendation system...")
    rec_engine = RecommendationEngine(method='content_based', n_recommendations=5)
    rec_engine.create_user_item_matrix(df)
    rec_engine.create_item_features()
    logger.info(f"    ✓ Recommendation system ready!")
    
    # 7. Make sample predictions
    logger.info("\n[7] Making sample predictions...")
    
    # Get a sample player
    sample_idx = 0
    sample_player = X_churn.iloc[sample_idx]
    
    # Churn prediction
    churn_prob = churn_model.predict(X_churn.iloc[[sample_idx]], return_probability=True)[0]
    logger.info(f"    ✓ Sample Player #{sample_idx + 1}")
    logger.info(f"      - Churn probability: {churn_prob:.2%}")
    
    # Fraud score
    fraud_score = fraud_model.predict(X_churn.iloc[[sample_idx]], return_scores=True)[0]
    logger.info(f"      - Fraud score: {fraud_score:.4f}")
    
    # Recommendations
    rec = rec_engine.recommend(user_id=sample_idx + 1, user_data=df)
    logger.info(f"      - Recommended items: {rec['recommended_items']}")
    
    # 8. Summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Start API: uvicorn src.api.main:app --reload")
    logger.info("  2. Run tests: pytest tests/ -v")
    logger.info("  3. Deploy with Docker: docker-compose up -d")
    logger.info("  4. View docs: http://localhost:8000/docs")
    logger.info("\nFor more info, see README.md")
    
    return {
        'df': df_features,
        'churn_model': churn_model,
        'fraud_model': fraud_model,
        'rec_engine': rec_engine,
        'metrics': {
            'churn': metrics,
            'fraud': fraud_metrics
        }
    }


if __name__ == "__main__":
    results = quick_demo()
