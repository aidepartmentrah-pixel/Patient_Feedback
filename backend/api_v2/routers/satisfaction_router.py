"""
Satisfaction Router
Endpoints for managing patient/case satisfaction data.

Used by:
- Worker and Complaint Administrator to add satisfaction to cases
- Patient History page to view satisfaction per case
"""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.api.db_layer import satisfaction_db
from backend.api.dependencies.user_context import get_current_user
from backend.api.schemas.auth_models import CurrentUser


router = APIRouter(prefix="/api/v2", tags=["satisfaction"])


# ============================================================
# MODELS
# ============================================================

class SatisfactionCreate(BaseModel):
    """Request model for creating satisfaction"""
    feedback_needed: bool
    feedback_given: bool
    feedback_datetime: Optional[str] = None  # ISO format datetime
    feedback_text: Optional[str] = None  # Feedback text/notes
    satisfaction_status_id: int  # 1=Not Present, 2=Satisfied, 3=Not Satisfied


class SatisfactionResponse(BaseModel):
    """Response model for satisfaction data"""
    satisfaction_id: int
    incident_case_id: int
    feedback_needed: bool
    feedback_given: bool
    feedback_datetime: Optional[str]
    satisfaction_status_id: int
    satisfaction_status_name_en: str
    satisfaction_status_name_ar: str
    created_by_user_id: int
    created_at: str


# ============================================================
# ENDPOINTS
# ============================================================

@router.get("/lookup/satisfaction-statuses")
async def get_satisfaction_statuses(
    current_user: CurrentUser = Depends(get_current_user)
) -> list:
    """
    Get all satisfaction status lookup values.
    
    Returns:
        [{ satisfaction_status_id, status_name_en, status_name_ar }, ...]
    """
    return satisfaction_db.get_satisfaction_statuses()


@router.get("/cases/{case_id}/satisfaction")
async def get_case_satisfaction(
    case_id: int,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get satisfaction record for a specific case.
    
    Returns:
        { satisfaction data } or { exists: false } if not found
    """
    satisfaction = satisfaction_db.get_satisfaction_by_case(case_id)
    
    if satisfaction:
        return {
            "exists": True,
            **satisfaction
        }
    else:
        return {
            "exists": False,
            "incident_case_id": case_id
        }


@router.post("/cases/{case_id}/satisfaction")
async def create_case_satisfaction(
    case_id: int,
    data: SatisfactionCreate,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create satisfaction record for a case.
    
    Allowed roles:
    - WORKER
    - COMPLAINT_SUPERVISOR
    - SOFTWARE_ADMIN
    
    Returns:
        { success: true, satisfaction_id: int }
        
    Raises:
        400: If satisfaction already exists for this case
        403: If user not authorized
    """
    # Check role authorization
    if not current_user or not current_user.scopes:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    role_code = current_user.scopes[0].role_code
    allowed_roles = {'WORKER', 'COMPLAINT_SUPERVISOR', 'SOFTWARE_ADMIN'}
    
    if role_code not in allowed_roles:
        raise HTTPException(
            status_code=403, 
            detail=f"Role {role_code} is not authorized to add satisfaction. Allowed: {', '.join(allowed_roles)}"
        )
    
    # Check if satisfaction already exists
    if satisfaction_db.satisfaction_exists(case_id):
        raise HTTPException(
            status_code=409, 
            detail="Satisfaction already exists for this case. Cannot edit after submission."
        )
    
    # Parse datetime if provided - if date only, append 12:00 PM
    feedback_dt = None
    if data.feedback_datetime:
        try:
            dt_str = data.feedback_datetime.replace('Z', '+00:00')
            # If it's just a date (YYYY-MM-DD), append 12:00 PM
            if len(dt_str) == 10 and '-' in dt_str:
                dt_str = f"{dt_str}T12:00:00"
            feedback_dt = datetime.fromisoformat(dt_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")
    
    # Validate status ID
    if data.satisfaction_status_id not in [1, 2, 3]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid satisfaction_status_id. Must be 1 (Not Present), 2 (Satisfied), or 3 (Not Satisfied)"
        )
    
    try:
        satisfaction_id = satisfaction_db.create_satisfaction(
            incident_case_id=case_id,
            feedback_needed=data.feedback_needed,
            feedback_given=data.feedback_given,
            feedback_datetime=feedback_dt,
            feedback_text=data.feedback_text,
            satisfaction_status_id=data.satisfaction_status_id,
            created_by_user_id=current_user.user_id
        )
        
        return {
            "success": True,
            "satisfaction_id": satisfaction_id,
            "message": "Satisfaction recorded successfully"
        }
    except Exception as e:
        if "already exists" in str(e).lower():
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to create satisfaction: {str(e)}")


@router.put("/cases/{case_id}/satisfaction")
async def update_case_satisfaction(
    case_id: int,
    data: SatisfactionCreate,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Update satisfaction record for a case.
    
    Allowed roles:
    - WORKER
    - COMPLAINT_SUPERVISOR
    - SOFTWARE_ADMIN
    
    Returns:
        { success: true, message: str }
        
    Raises:
        404: If satisfaction not found for this case
        403: If user not authorized
    """
    # Check role authorization
    if not current_user or not current_user.scopes:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    role_code = current_user.scopes[0].role_code
    allowed_roles = {'WORKER', 'COMPLAINT_SUPERVISOR', 'SOFTWARE_ADMIN'}
    
    if role_code not in allowed_roles:
        raise HTTPException(
            status_code=403, 
            detail=f"Role {role_code} is not authorized to update satisfaction. Allowed: {', '.join(allowed_roles)}"
        )
    
    # Check if satisfaction exists
    if not satisfaction_db.satisfaction_exists(case_id):
        raise HTTPException(
            status_code=404, 
            detail="Satisfaction not found for this case"
        )
    
    # Parse datetime if provided - if date only, append 12:00 PM
    feedback_dt = None
    if data.feedback_datetime:
        try:
            dt_str = data.feedback_datetime.replace('Z', '+00:00')
            # If it's just a date (YYYY-MM-DD), append 12:00 PM
            if len(dt_str) == 10 and '-' in dt_str:
                dt_str = f"{dt_str}T12:00:00"
            feedback_dt = datetime.fromisoformat(dt_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid datetime format")
    
    # Validate status ID
    if data.satisfaction_status_id not in [1, 2, 3]:
        raise HTTPException(
            status_code=400, 
            detail="Invalid satisfaction_status_id. Must be 1 (Not Present), 2 (Satisfied), or 3 (Not Satisfied)"
        )
    
    try:
        updated = satisfaction_db.update_satisfaction(
            incident_case_id=case_id,
            feedback_needed=data.feedback_needed,
            feedback_given=data.feedback_given,
            feedback_datetime=feedback_dt,
            feedback_text=data.feedback_text,
            satisfaction_status_id=data.satisfaction_status_id,
            modified_by_user_id=current_user.user_id
        )
        
        if updated:
            return {
                "success": True,
                "message": "Satisfaction updated successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Satisfaction not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update satisfaction: {str(e)}")
