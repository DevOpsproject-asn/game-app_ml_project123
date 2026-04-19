"""MLflow integration service"""
import mlflow
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MLflowService:
    """Service for MLflow model tracking."""
    
    def __init__(self, tracking_uri="http://localhost:5000", experiment_name="game-ml"):
        """Initialize MLflow service."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow initialized with URI: {tracking_uri}")
    
    def log_run(self, run_name: str, params: Dict[str, Any], metrics: Dict[str, float]):
        """Log a training run."""
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            
            # Log metrics
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)
            
            logger.info(f"Run logged: {run_name}")
    
    def log_model(self, model, artifact_path: str, model_name: str):
        """Log model artifact."""
        mlflow.sklearn.log_model(model, artifact_path=artifact_path)
        logger.info(f"Model logged: {model_name}")
    
    def get_best_run(self, metric: str = "f1"):
        """Get best run by metric."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            if not runs.empty:
                best_run = runs.loc[runs[f'metrics.{metric}'].idxmax()]
                return best_run
        return None
    
    def list_experiments(self):
        """List all experiments."""
        experiments = mlflow.search_experiments()
        return [exp.name for exp in experiments]
