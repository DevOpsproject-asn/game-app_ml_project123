"""Routes for recommendations"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class RecommendationRequest(BaseModel):
    """Recommendation request."""
    user_id: int
    liked_items: List[int] = []
    n_recommendations: int = 5


class RecommendationResponse(BaseModel):
    """Recommendation response."""
    user_id: int
    recommended_items: List[int]
    method: str
    confidence: str


@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get item/level recommendations for a player.
    
    Returns:
        - recommended_items: List of item IDs
        - method: 'content_based' or 'collaborative'
        - confidence: 'high' or 'low'
    """
    try:
        # Simulate recommendation (in production, use RecommendationEngine)
        import random
        
        # Filter liked items from recommendations
        all_items = set(range(1, 101))  # Assume 100 items
        available_items = list(all_items - set(request.liked_items))
        
        # Get recommendations
        recommendations = random.sample(
            available_items, 
            min(request.n_recommendations, len(available_items))
        )
        
        confidence = "high" if len(recommendations) >= request.n_recommendations else "low"
        
        logger.info(f"Recommendations for user {request.user_id}: {recommendations}")
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommended_items=recommendations,
            method="content_based",
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
