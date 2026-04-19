"""Routes for churn prediction"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class PlayerData(BaseModel):
    """Player data model."""
    user_id: int
    session_time: int
    levels_completed: int
    in_game_purchases: int
    last_login_days_ago: int
    total_sessions: int
    daily_active: int
    player_level: int
    total_playtime_hours: float
    achievement_count: int
    is_premium: int


class ChurnPredictionRequest(BaseModel):
    """Churn prediction request."""
    player_data: PlayerData


class ChurnPredictionResponse(BaseModel):
    """Churn prediction response."""
    user_id: int
    churn_probability: float
    risk_level: str
    recommendation: str


@router.post("/predict-churn", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """
    Predict churn probability for a player.
    
    Returns:
        - churn_probability: Float between 0 and 1
        - risk_level: 'high', 'medium', 'low'
        - recommendation: Action to take
    """
    try:
        player = request.player_data
        
        # Simulate churn prediction (in production, use trained model)
        # Factors: inactivity, low engagement, low spend
        inactivity_score = min(player.last_login_days_ago / 90, 1.0)
        engagement_score = min(player.total_sessions / 100, 1.0)
        value_score = min(player.in_game_purchases / 50, 1.0)
        
        churn_probability = 0.4 * inactivity_score + 0.3 * (1 - engagement_score) + 0.3 * (1 - value_score)
        churn_probability = min(max(churn_probability, 0), 1)
        
        # Risk level
        if churn_probability > 0.7:
            risk_level = "high"
            recommendation = "Send special offer/reward, increase engagement"
        elif churn_probability > 0.4:
            risk_level = "medium"
            recommendation = "Personalized content/recommendations"
        else:
            risk_level = "low"
            recommendation = "Maintain engagement"
        
        logger.info(f"Churn prediction for user {player.user_id}: {churn_probability:.2%}")
        
        return ChurnPredictionResponse(
            user_id=player.user_id,
            churn_probability=churn_probability,
            risk_level=risk_level,
            recommendation=recommendation
        )
    
    except Exception as e:
        logger.error(f"Churn prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict-churn-batch")
async def predict_churn_batch(players: List[PlayerData]):
    """Predict churn for multiple players."""
    results = []
    
    for player in players:
        request = ChurnPredictionRequest(player_data=player)
        result = await predict_churn(request)
        results.append(result)
    
    logger.info(f"Batch churn prediction for {len(players)} players")
    return {"predictions": results, "count": len(results)}
