"""
Performance Scoring Module
Pure logic module for computing performance scores from metrics.

This module provides centralized scoring rules for converting worker/doctor
performance metrics into numerical scores with qualitative labels and risk flags.

Design Principles:
    - Pure logic module (no database, no FastAPI dependencies)
    - Deterministic scoring (same inputs always produce same output)
    - Reusable by both worker and doctor reporting systems
    - Adjustable policy layer (scoring rules can be modified here)

Note:
    Scoring rules defined here represent organizational policy and can be
    adjusted without changing service or router layers.
"""

from typing import List
from pydantic import BaseModel, Field


class PerformanceScoreResult(BaseModel):
    """
    Result of performance score computation.
    
    Contains numerical score, qualitative assessment labels, and risk flags.
    
    Attributes:
        score: Numerical performance score (0-100, higher is better)
        praise_level: Qualitative assessment ("excellent", "good", "watch", "critical")
        risk_level: Risk assessment ("low", "medium", "high")
        flags: List of warning flags for specific concerns
    
    Example:
        >>> result = PerformanceScoreResult(
        ...     score=82,
        ...     praise_level="good",
        ...     risk_level="low",
        ...     flags=[]
        ... )
    """
    score: int = Field(..., ge=0, le=100, description="Performance score (0-100)")
    praise_level: str = Field(..., description="Qualitative assessment level")
    risk_level: str = Field(..., description="Risk level assessment")
    flags: List[str] = Field(default_factory=list, description="Warning flags")
    
    class Config:
        json_schema_extra = {
            "example": {
                "score": 82,
                "praise_level": "good",
                "risk_level": "low",
                "flags": []
            }
        }


def compute_performance_score(
    total_incidents: int = 0,
    completed_actions: int = 0,
    overdue_actions: int = 0,
    rejected_explanations: int = 0
) -> PerformanceScoreResult:
    """
    Compute performance score from worker/doctor metrics.
    
    This function implements organizational policy for converting raw performance
    metrics into a scored assessment with labels and flags. The scoring algorithm
    is deterministic and adjustable.
    
    Scoring Algorithm:
        1. Start with base score of 100
        2. Apply penalties for negative signals:
           - Overdue actions: -5 points each
           - Rejected explanations: -7 points each
           - Total incidents: -1 point each
        3. Apply bonuses for positive signals:
           - Completed actions: +2 points each
        4. Clamp final score to range [0, 100]
    
    Praise Level Classification:
        - score >= 85: "excellent" (high performer)
        - score >= 70: "good" (solid performer)
        - score >= 50: "watch" (needs attention)
        - score < 50: "critical" (immediate intervention needed)
    
    Risk Level Classification:
        - "high": overdue > 5 OR rejected > 3 (serious risk)
        - "medium": overdue > 2 (moderate risk)
        - "low": otherwise (minimal risk)
    
    Warning Flags:
        - "many_overdue": overdue_actions > 5
        - "many_rejections": rejected_explanations > 3
    
    Policy Adjustability:
        Scoring weights and thresholds can be modified here to reflect
        changing organizational priorities without touching service/router layers.
    
    Args:
        total_incidents: Total number of incidents involving person
        completed_actions: Number of completed action items
        overdue_actions: Number of overdue action items
        rejected_explanations: Number of rejected explanations
    
    Returns:
        PerformanceScoreResult with score, labels, and flags
    
    Example:
        >>> # High performer
        >>> result = compute_performance_score(
        ...     total_incidents=5,
        ...     completed_actions=20,
        ...     overdue_actions=0,
        ...     rejected_explanations=0
        ... )
        >>> print(f"Score: {result.score}, Level: {result.praise_level}")
        Score: 135, Level: excellent  # Note: will be clamped to 100
        
        >>> # Poor performer
        >>> result = compute_performance_score(
        ...     total_incidents=15,
        ...     completed_actions=2,
        ...     overdue_actions=8,
        ...     rejected_explanations=5
        ... )
        >>> print(f"Score: {result.score}, Flags: {result.flags}")
        Score: 0, Flags: ['many_overdue', 'many_rejections']
    """
    # ============================================
    # STEP 1: COMPUTE RAW SCORE
    # ============================================
    
    # Start with perfect score
    base_score = 100
    
    # Apply penalties (negative signals)
    penalty_overdue = overdue_actions * 5
    penalty_rejected = rejected_explanations * 7
    penalty_incidents = total_incidents * 1
    
    # Apply bonuses (positive signals)
    bonus_completed = completed_actions * 2
    
    # Calculate raw score
    raw_score = (
        base_score
        - penalty_overdue
        - penalty_rejected
        - penalty_incidents
        + bonus_completed
    )
    
    # Clamp score to valid range [0, 100]
    score = max(0, min(100, raw_score))
    
    # ============================================
    # STEP 2: DETERMINE PRAISE LEVEL
    # ============================================
    
    if score >= 85:
        praise_level = "excellent"
    elif score >= 70:
        praise_level = "good"
    elif score >= 50:
        praise_level = "watch"
    else:
        praise_level = "critical"
    
    # ============================================
    # STEP 3: DETERMINE RISK LEVEL
    # ============================================
    
    if overdue_actions > 5 or rejected_explanations > 3:
        risk_level = "high"
    elif overdue_actions > 2:
        risk_level = "medium"
    else:
        risk_level = "low"
    
    # ============================================
    # STEP 4: COLLECT FLAGS
    # ============================================
    
    flags = []
    
    if overdue_actions > 5:
        flags.append("many_overdue")
    
    if rejected_explanations > 3:
        flags.append("many_rejections")
    
    # ============================================
    # STEP 5: BUILD RESULT
    # ============================================
    
    result = PerformanceScoreResult(
        score=score,
        praise_level=praise_level,
        risk_level=risk_level,
        flags=flags
    )
    
    return result
