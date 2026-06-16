"""
Publication Batch Router (API V2)
Read-only endpoints for HCAT Performance & Delay Monitoring (Stage 2),
Session 1 - Publication Batch Tracking.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List

from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser
from backend.api.utils.guards import require_logged_in
from backend.api_v2.db_layer import publication_batch_db


router = APIRouter(prefix="/api/publication-batches", tags=["Publication Batches"])


@router.get("/recent")
def get_recent_publication_batches(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return the latest 5 publication batches (most recent first).
    """
    require_logged_in(current_user)
    batches = publication_batch_db.get_recent_publication_batches(limit=5)
    return {"batches": batches}


@router.get("/{batch_id}/cases")
def get_publication_batch_cases(
    batch_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Return the cases included in a single publication batch.
    """
    require_logged_in(current_user)
    cases = publication_batch_db.get_publication_batch_cases(batch_id)
    return {"cases": cases}
