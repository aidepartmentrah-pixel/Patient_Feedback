"""
API v2 Routers Package
FastAPI routers with role-based and scope-based guards applied.

Note: Insight router has been formally delayed (see STEP_3_5_5_INSIGHT_DELAY_DECISION.md)
"""

from .workflow_router import router as workflow_router
from .action_log_router import router as action_log_router

__all__ = [
    "workflow_router",
    "action_log_router",
]
