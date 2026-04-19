"""FastAPI main application"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from src.api.routes import churn, recommendation, fraud, insights, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Application startup")
    yield
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Game ML Platform API",
    description="ML-powered gaming analytics and prediction API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/v1", tags=["Health"])
app.include_router(churn.router, prefix="/api/v1", tags=["Churn"])
app.include_router(recommendation.router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(fraud.router, prefix="/api/v1", tags=["Fraud Detection"])
app.include_router(insights.router, prefix="/api/v1", tags=["Player Insights"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Game ML Platform API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/v1/health",
            "churn": "/api/v1/predict-churn",
            "recommendations": "/api/v1/recommend",
            "fraud": "/api/v1/detect-fraud",
            "insights": "/api/v1/player-insights"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
