"""Routes for player insights"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class PlayerInsightRequest(BaseModel):
    """Player insight request."""
    user_id: int
    total_sessions: int
    levels_completed: int
    in_game_purchases: int
    last_login_days_ago: int
    total_playtime_hours: float
    achievement_count: int


class PlayerInsightResponse(BaseModel):
    """Player insight response."""
    user_id: int
    engagement_level: str
    lifetime_value_estimate: float
    retention_risk: str
    recommended_actions: list


@router.post("/player-insights", response_model=PlayerInsightResponse)
async def get_player_insights(request: PlayerInsightRequest):
    """
    Get comprehensive insights about a player.
    
    Returns:
        - engagement_level: 'low', 'medium', 'high'
        - lifetime_value_estimate: Estimated monetary value
        - retention_risk: 'low', 'medium', 'high'
        - recommended_actions: List of suggested actions
    """
    try:
        # Engagement level based on sessions and playtime
        engagement_score = (
            min(request.total_sessions / 100, 1.0) * 0.4 +
            min(request.total_playtime_hours / 50, 1.0) * 0.3 +
            min(request.achievement_count / 50, 1.0) * 0.3
        )
        
        if engagement_score > 0.6:
            engagement_level = "high"
        elif engagement_score > 0.3:
            engagement_level = "medium"
        else:
            engagement_level = "low"
        
        # Lifetime value estimate
        lifetime_value = (
            request.in_game_purchases * 10 +
            request.total_playtime_hours * 1 +
            request.achievement_count * 5
        )
        
        # Retention risk
        if request.last_login_days_ago > 30:
            retention_risk = "high"
        elif request.last_login_days_ago > 7:
            retention_risk = "medium"
        else:
            retention_risk = "low"
        
        # Recommended actions
        actions = []
        if retention_risk == "high":
            actions.append("Send re-engagement offer")
            actions.append("Send exclusive reward")
        if engagement_level == "high" and lifetime_value > 100:
            actions.append("Offer premium features")
            actions.append("Invite to VIP program")
        if engagement_level == "low":
            actions.append("Introduce tutorial/help")
            actions.append("Send beginner rewards")
        
        logger.info(f"Player insights for user {request.user_id}: engagement={engagement_level}, ltv=${lifetime_value:.2f}")
        
        return PlayerInsightResponse(
            user_id=request.user_id,
            engagement_level=engagement_level,
            lifetime_value_estimate=lifetime_value,
            retention_risk=retention_risk,
            recommended_actions=actions
        )
    
    except Exception as e:
        logger.error(f"Player insights error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
