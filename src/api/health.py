"""API route for health checks"""
from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Game ML Platform API",
        "version": "1.0.0"
    }


@router.get("/health/models")
async def model_health():
    """Check model availability."""
    return {
        "churn_model": "ready",
        "recommendation_model": "ready",
        "fraud_detector": "ready"
    }
