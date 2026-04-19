"""Routes for fraud detection"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


class FraudCheckRequest(BaseModel):
    """Fraud detection request."""
    user_id: int
    session_time: int
    levels_completed: int
    in_game_purchases: int
    total_playtime_hours: float
    player_level: int


class FraudCheckResponse(BaseModel):
    """Fraud detection response."""
    user_id: int
    fraud_score: float
    is_fraud: bool
    risk_level: str
    flags: List[str]


@router.post("/detect-fraud", response_model=FraudCheckResponse)
async def detect_fraud(request: FraudCheckRequest):
    """
    Detect fraudulent/bot player activity.
    
    Returns:
        - fraud_score: Float between 0 and 1
        - is_fraud: Boolean flag
        - risk_level: 'high', 'medium', 'low'
        - flags: List of detected anomalies
    """
    try:
        flags = []
        
        # Rule 1: Session too long (impossible for human player)
        if request.session_time > 720:  # 12 hours
            flags.append("Impossible session duration")
        
        # Rule 2: Unrealistic progression
        if request.total_playtime_hours > 0:
            levels_per_hour = request.levels_completed / request.total_playtime_hours
            if levels_per_hour > 50:
                flags.append("Unrealistic level progression speed")
        
        # Rule 3: Suspicious purchases
        if request.player_level > 0:
            purchase_per_level = request.in_game_purchases / request.player_level
            if purchase_per_level > 10:
                flags.append("Suspicious purchase pattern")
        
        # Calculate fraud score
        fraud_score = 0.0
        if len(flags) > 0:
            fraud_score = min(0.25 * len(flags), 1.0)
        
        # Determine if flagged as fraud
        is_fraud = fraud_score > 0.5
        
        if fraud_score > 0.7:
            risk_level = "high"
        elif fraud_score > 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        logger.info(f"Fraud check for user {request.user_id}: score={fraud_score:.2f}, flags={flags}")
        
        return FraudCheckResponse(
            user_id=request.user_id,
            fraud_score=fraud_score,
            is_fraud=is_fraud,
            risk_level=risk_level,
            flags=flags
        )
    
    except Exception as e:
        logger.error(f"Fraud detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
