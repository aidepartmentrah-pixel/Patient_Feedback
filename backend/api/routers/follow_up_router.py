"""
Follow-Up Actions Router
API endpoints for follow-up action management.

⚠️ DEPRECATION WARNING — API v1 LEGACY ENDPOINT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This router is DEPRECATED and should NOT be used for new frontend development.

API v2 Replacement:
- Use /api/v2/follow-up/action-items endpoints instead
- API v2 provides proper role-based access control and scope enforcement
- This router lacks security guards and should be considered INTERNAL only

Status:
- Maintained for backward compatibility with existing integrations
- Will be removed or disabled after Phase 4 frontend migration
- Contains 12+ endpoints that duplicate API v2 functionality

Security Risk:
- No role guards (any authenticated user can access)
- No scope enforcement (can access any action item)
- No ownership validation (can modify any action)

TODO: Remove or disable after Phase 4 migration
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from fastapi import APIRouter, Query, Path, HTTPException, Depends
from typing import Optional
from pydantic import BaseModel

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in
from ..services.follow_up_service import FollowUpService


router = APIRouter(prefix="/api/follow-up", tags=["Follow-Up Actions"])


# ==================== REQUEST/RESPONSE MODELS ====================

class ActionResponse(BaseModel):
    """Response model for follow-up action."""
    id: int
    actionTitle: str
    actionDescription: Optional[str]
    sourceType: str
    sourceId: str
    departmentId: Optional[int]
    assignedTo: Optional[str]
    priority: str
    status: str
    dueDate: str
    completedDate: Optional[str]
    notes: Optional[str]
    createdAt: str
    createdByUserId: int
    lastUpdatedAt: Optional[str]
    lastUpdatedByUserId: Optional[int]
    isOverdue: bool
    daysRemaining: int
    daysOverdue: int


class UpdateActionRequest(BaseModel):
    """Request model for updating action."""
    dueDate: Optional[str] = None
    assignedTo: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class CompleteActionRequest(BaseModel):
    """Request model for completing action."""
    completionNotes: Optional[str] = None
    completedDate: Optional[str] = None


class DelayActionRequest(BaseModel):
    """Request model for delaying action."""
    delayDays: int
    reason: Optional[str] = None


class CreateActionRequest(BaseModel):
    """Request model for creating action."""
    actionTitle: str
    actionDescription: Optional[str] = None
    incidentCaseId: Optional[int] = None
    seasonalReportId: Optional[int] = None
    departmentId: Optional[int] = None
    assignedTo: Optional[str] = None
    priority: str = 'medium'
    dueDate: str
    notes: Optional[str] = None


class BulkCompleteRequest(BaseModel):
    """Request model for bulk completing actions."""
    actionIds: list[int]
    completionNotes: Optional[str] = None
    completedDate: Optional[str] = None


class BulkDelayRequest(BaseModel):
    """Request model for bulk delaying actions."""
    actionIds: list[int]
    delayDays: int
    reason: Optional[str] = None


class BulkUpdateRequest(BaseModel):
    """Request model for bulk updating actions."""
    actionIds: list[int]
    assignedTo: Optional[str] = None
    priority: Optional[str] = None
    departmentId: Optional[int] = None


# ==================== PHASE 3: CREATE ACTION ====================

@router.post("/actions")
async def create_follow_up_action(
    request: CreateActionRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Create a new follow-up action.
    
    **Request Body Example:**
    ```json
    {
      "actionTitle": "تدريب الطاقم على البروتوكول الجديد",
      "actionDescription": "تنفيذ دورة تدريبية شاملة",
      "incidentCaseId": 123,
      "departmentId": 5,
      "assignedTo": "د. أحمد علي",
      "priority": "high",
      "dueDate": "2026-02-15",
      "notes": "يجب إكمال التدريب قبل نهاية الشهر"
    }
    ```
    
    **Validation Rules:**
    - `actionTitle` is required
    - `dueDate` is required (YYYY-MM-DD format)
    - `priority` must be: high, medium, or low
    - Can link to incident OR seasonal report (not both)
    - If no source, action is standalone (manual)
    
    **Response:** Created action with generated ID
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        created_action = FollowUpService.create_follow_up_action(
            action_title=request.actionTitle,
            action_description=request.actionDescription,
            incident_case_id=request.incidentCaseId,
            seasonal_report_id=request.seasonalReportId,
            department_id=request.departmentId,
            assigned_to=request.assignedTo,
            priority=request.priority,
            due_date=request.dueDate,
            notes=request.notes,
            user_id=user_id
        )
        return created_action
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTION_CREATION_FAILED",
            "message": str(e),
            "message_ar": f"فشل إنشاء الإجراء: {str(e)}"
        })


# ==================== B.1: GET FOLLOW-UP ACTIONS ====================

@router.get("/actions")
async def get_follow_up_actions(
    status: Optional[str] = Query(None, description="Filter by status: pending, delayed, completed, or all"),
    priority: Optional[str] = Query(None, description="Filter by priority: high, medium, low, or all"),
    department: Optional[str] = Query(None, description="Filter by department ID or 'all'"),
    from_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    to_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    include_completed: bool = Query(False, description="Include completed actions (default: false)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get follow-up actions with optional filtering and sorting.
    
    Results sorted by due date (earliest first).
    Includes derived fields: isOverdue, daysRemaining, daysOverdue.
    Statistics are global (not affected by filters).
    
    **Query Parameters:**
    - `status`: pending, delayed, completed, or 'all'
    - `priority`: high, medium, low, or 'all'
    - `department`: Department ID or 'all'
    - `from_date`: Filter actions due from this date (YYYY-MM-DD)
    - `to_date`: Filter actions due until this date (YYYY-MM-DD)
    - `include_completed`: Include completed actions (default: false)
    
    **Response:**
    ```json
    {
      "actions": [
        {
          "id": 1,
          "actionTitle": "تدريب الطاقم",
          "sourceType": "incident_explanation",
          "sourceId": "INC-001",
          "status": "pending",
          "priority": "high",
          "dueDate": "2026-01-15",
          "isOverdue": false,
          "daysRemaining": 9
        }
      ],
      "total": 45,
      "statistics": {
        "actionsToTake": 32,
        "overdue": 8,
        "completed": 13
      }
    }
    ```
    """
    require_logged_in(current_user)
    
    try:
        result = FollowUpService.get_follow_up_actions(
            status=status,
            priority=priority,
            department=department,
            from_date=from_date,
            to_date=to_date,
            include_completed=include_completed
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTIONS_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب إجراءات المتابعة: {str(e)}"
        })


# ==================== B.2: GET SINGLE ACTION ====================

@router.get("/actions/{action_id}")
async def get_follow_up_action_by_id(
    action_id: int = Path(..., gt=0, description="Action ID"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get detailed information for a specific follow-up action.
    
    **Path Parameters:**
    - `action_id`: Action unique identifier
    
    **Response:**
    ```json
    {
      "id": 1,
      "actionTitle": "تدريب الطاقم على بروتوكول جديد",
      "actionDescription": "إجراء دورة تدريبية لجميع الممرضين",
      "sourceType": "incident_explanation",
      "sourceId": "INC-001",
      "departmentId": 5,
      "assignedTo": "د. أحمد علي",
      "priority": "high",
      "status": "pending",
      "dueDate": "2026-01-15",
      "completedDate": null,
      "notes": "[2026-01-06 10:30] (user_id=1): Initial creation",
      "createdAt": "2026-01-06T10:30:00",
      "createdByUserId": 1,
      "lastUpdatedAt": "2026-01-06T10:30:00",
      "lastUpdatedByUserId": 1,
      "isOverdue": false,
      "daysRemaining": 9,
      "daysOverdue": 0
    }
    ```
    """
    require_logged_in(current_user)
    
    try:
        action = FollowUpService.get_follow_up_action_by_id(action_id)
        return action
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail={
            "error": "ACTION_NOT_FOUND",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTION_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب الإجراء: {str(e)}"
        })


# ==================== B.3: UPDATE ACTION ====================

@router.patch("/actions/{action_id}")
async def update_follow_up_action(
    action_id: int = Path(..., gt=0, description="Action ID"),
    request: UpdateActionRequest = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update follow-up action (partial update with status validation).
    
    Only send fields that have changed.
    Backend validates status transitions.
    Notes are appended (never overwritten).
    
    **Path Parameters:**
    - `action_id`: Action unique identifier
    
    **Request Body Example (Delay action):**
    ```json
    {
      "dueDate": "2026-01-22",
      "notes": "تأجيل لمدة أسبوع لانتظار موافقة الإدارة"
    }
    ```
    
    **Request Body Example (Reassign):**
    ```json
    {
      "assignedTo": "د. خالد محمد"
    }
    ```
    
    **Request Body Example (Change priority):**
    ```json
    {
      "priority": "high",
      "notes": "تم رفع الأولوية"
    }
    ```
    
    **Status Transition Rules:**
    - pending → delayed, completed
    - delayed → pending, completed
    - completed → (no further changes)
    
    **Response:** Updated action object (same as GET /actions/{id})
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context (would come from auth middleware)
        user_id = 1  # TODO: Get from authenticated request
        
        updated_action = FollowUpService.update_follow_up_action(
            action_id=action_id,
            due_date=request.dueDate,
            assigned_to=request.assignedTo,
            priority=request.priority,
            status=request.status,
            notes=request.notes,
            user_id=user_id
        )
        return updated_action
    
    except ValueError as e:
        error_msg = str(e)
        status_code = 404 if "not found" in error_msg.lower() else 400
        raise HTTPException(status_code=status_code, detail={
            "error": "INVALID_REQUEST" if status_code == 400 else "ACTION_NOT_FOUND",
            "message": error_msg,
            "message_ar": error_msg
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTION_UPDATE_FAILED",
            "message": str(e),
            "message_ar": f"فشل تحديث الإجراء: {str(e)}"
        })


# ==================== B.4: COMPLETE ACTION ====================

@router.post("/actions/{action_id}/complete")
async def complete_follow_up_action(
    action_id: int = Path(..., gt=0, description="Action ID"),
    request: CompleteActionRequest = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Mark follow-up action as completed.
    
    Sets status to 'completed' (final state, cannot reopen).
    Appends completion notes automatically.
    
    **Path Parameters:**
    - `action_id`: Action unique identifier
    
    **Request Body Example:**
    ```json
    {
      "completionNotes": "تم إنجاز التدريب بنجاح في 2026-01-15",
      "completedDate": "2026-01-15"
    }
    ```
    
    **Response:** Updated action with status='completed'
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        request_obj = request or CompleteActionRequest()
        
        completed_action = FollowUpService.complete_follow_up_action(
            action_id=action_id,
            completion_notes=request_obj.completionNotes,
            completed_date=request_obj.completedDate,
            user_id=user_id
        )
        return completed_action
    
    except ValueError as e:
        error_msg = str(e)
        status_code = 404 if "not found" in error_msg.lower() else 400
        raise HTTPException(status_code=status_code, detail={
            "error": "INVALID_REQUEST" if status_code == 400 else "ACTION_NOT_FOUND",
            "message": error_msg,
            "message_ar": error_msg
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTION_COMPLETION_FAILED",
            "message": str(e),
            "message_ar": f"فشل إنجاز الإجراء: {str(e)}"
        })


# ==================== B.5: DELAY ACTION ====================

@router.post("/actions/{action_id}/delay")
async def delay_follow_up_action(
    action_id: int = Path(..., gt=0, description="Action ID"),
    request: DelayActionRequest = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Delay follow-up action by specified number of days.
    
    Automatically updates dueDate and appends reason to notes.
    Sets status back to 'pending' if it was 'delayed'.
    
    **Path Parameters:**
    - `action_id`: Action unique identifier
    
    **Request Body Example:**
    ```json
    {
      "delayDays": 7,
      "reason": "انتظار موافقة الإدارة"
    }
    ```
    
    **Response:** Updated action with new dueDate
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        if not request or request.delayDays <= 0:
            raise ValueError("delayDays must be a positive number")
        
        delayed_action = FollowUpService.delay_follow_up_action(
            action_id=action_id,
            delay_days=request.delayDays,
            reason=request.reason,
            user_id=user_id
        )
        return delayed_action
    
    except ValueError as e:
        error_msg = str(e)
        status_code = 404 if "not found" in error_msg.lower() else 400
        raise HTTPException(status_code=status_code, detail={
            "error": "INVALID_REQUEST" if status_code == 400 else "ACTION_NOT_FOUND",
            "message": error_msg,
            "message_ar": error_msg
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTION_DELAY_FAILED",
            "message": str(e),
            "message_ar": f"فشل تأجيل الإجراء: {str(e)}"
        })


# ==================== PHASE 2: REOPEN ACTION ====================

class ReopenActionRequest(BaseModel):
    """Request model for reopening action."""
    reopenReason: str
    newDueDate: Optional[str] = None


@router.post("/actions/{action_id}/reopen")
async def reopen_follow_up_action(
    action_id: int = Path(..., gt=0, description="Action ID"),
    request: ReopenActionRequest = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Reopen a completed action back to pending status.
    
    Allows reopening completed actions that were marked complete by mistake
    or need to be re-activated. Requires a reason for audit purposes.
    
    **Path Parameters:**
    - `action_id`: Action unique identifier
    
    **Request Body Example:**
    ```json
    {
      "reopenReason": "تم الإغلاق بالخطأ - يحتاج إعادة متابعة",
      "newDueDate": "2026-01-20"
    }
    ```
    
    **Response:** Updated action with status='pending'
    
    **Notes:**
    - Only completed actions can be reopened
    - Reason is required for audit trail
    - If newDueDate not provided, defaults to today
    - CompletedDate is cleared
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        if not request or not request.reopenReason or not request.reopenReason.strip():
            raise ValueError("reopenReason is required")
        
        reopened_action = FollowUpService.reopen_follow_up_action(
            action_id=action_id,
            reopen_reason=request.reopenReason,
            new_due_date=request.newDueDate,
            user_id=user_id
        )
        return reopened_action
    
    except ValueError as e:
        error_msg = str(e)
        status_code = 404 if "not found" in error_msg.lower() else 400
        raise HTTPException(status_code=status_code, detail={
            "error": "INVALID_REQUEST" if status_code == 400 else "ACTION_NOT_FOUND",
            "message": error_msg,
            "message_ar": error_msg
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "ACTION_REOPEN_FAILED",
            "message": str(e),
            "message_ar": f"فشل إعادة فتح الإجراء: {str(e)}"
        })


# ==================== PHASE 2: ACTION HISTORY ====================

@router.get("/actions/{action_id}/history")
async def get_action_history(
    action_id: int = Path(..., gt=0, description="Action ID"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get change history for a follow-up action.
    
    Returns parsed timeline of all changes made to the action
    based on timestamped notes entries.
    
    **Path Parameters:**
    - `action_id`: Action unique identifier
    
    **Response Example:**
    ```json
    {
      "actionId": 1,
      "history": [
        {
          "timestamp": "2026-01-06 10:30",
          "userId": 1,
          "action": "Initial creation",
          "details": "Initial creation"
        },
        {
          "timestamp": "2026-01-07 14:15",
          "userId": 2,
          "action": "Delayed 7 days",
          "details": "Delayed 7 days - waiting for approval"
        },
        {
          "timestamp": "2026-01-10 09:00",
          "userId": 1,
          "action": "Action marked complete",
          "details": "Action marked complete"
        }
      ]
    }
    ```
    
    **Notes:**
    - History extracted from notes field
    - Chronological order (oldest first)
    - Only includes properly formatted entries
    """
    require_logged_in(current_user)
    
    try:
        result = FollowUpService.get_action_history(action_id)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail={
            "error": "ACTION_NOT_FOUND",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "HISTORY_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب التاريخ: {str(e)}"
        })


# ==================== PHASE 2: CALENDAR VIEW ====================

@router.get("/calendar")
async def get_calendar_actions(
    year: int = Query(..., ge=2000, le=2100, description="Year (2000-2100)"),
    month: int = Query(..., ge=1, le=12, description="Month (1-12)"),
    department: Optional[str] = Query(None, description="Filter by department ID or 'all'"),
    status: Optional[str] = Query(None, description="Filter by status or 'all' (default: pending+delayed only)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get actions grouped by date for calendar visualization.
    
    Returns actions organized by due date for easy calendar rendering.
    By default, excludes completed actions.
    
    **Query Parameters:**
    - `year`: Year (2000-2100)
    - `month`: Month (1-12)
    - `department`: Department ID or 'all' (optional)
    - `status`: Status filter or 'all' (optional, default excludes completed)
    
    **Response Example:**
    ```json
    {
      "year": 2026,
      "month": 1,
      "calendar": {
        "2026-01-15": [
          {
            "id": 1,
            "actionTitle": "تدريب الطاقم",
            "priority": "high",
            "status": "pending",
            "departmentId": 5,
            "assignedTo": "د. أحمد",
            "isOverdue": false
          }
        ],
        "2026-01-20": [
          {
            "id": 2,
            "actionTitle": "مراجعة السياسات",
            "priority": "medium",
            "status": "pending",
            "departmentId": 3,
            "assignedTo": "د. فاطمة",
            "isOverdue": false
          }
        ]
      }
    }
    ```
    
    **Notes:**
    - Actions sorted by priority within each date
    - Only pending/delayed actions shown by default
    - Perfect for calendar/timeline UI components
    """
    require_logged_in(current_user)
    
    try:
        result = FollowUpService.get_calendar_actions(
            year=year,
            month=month,
            department=department,
            status=status
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "CALENDAR_FETCH_FAILED",
            "message": str(e),
            "message_ar": f"فشل جلب التقويم: {str(e)}"
        })


# ==================== PHASE 3: BULK COMPLETE ====================

@router.post("/actions/bulk-complete")
async def bulk_complete_actions(
    request: BulkCompleteRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Mark multiple actions as completed in one operation.
    
    **Request Body Example:**
    ```json
    {
      "actionIds": [1, 2, 3, 5],
      "completionNotes": "تم إنجاز جميع الإجراءات في الاجتماع الأسبوعي",
      "completedDate": "2026-01-07"
    }
    ```
    
    **Response Example:**
    ```json
    {
      "successCount": 3,
      "failedCount": 1,
      "failedIds": [
        {
          "id": 5,
          "reason": "Action not found"
        }
      ]
    }
    ```
    
    **Notes:**
    - Skips already completed actions
    - Returns detailed results for each action
    - All successful updates committed as single transaction
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        result = FollowUpService.bulk_complete_actions(
            action_ids=request.actionIds,
            completion_notes=request.completionNotes,
            completed_date=request.completedDate,
            user_id=user_id
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "BULK_COMPLETE_FAILED",
            "message": str(e),
            "message_ar": f"فشل الإنجاز الجماعي: {str(e)}"
        })


# ==================== PHASE 3: BULK DELAY ====================

@router.post("/actions/bulk-delay")
async def bulk_delay_actions(
    request: BulkDelayRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Delay multiple actions by specified days in one operation.
    
    **Request Body Example:**
    ```json
    {
      "actionIds": [1, 2, 3],
      "delayDays": 7,
      "reason": "انتظار موافقة الإدارة العليا"
    }
    ```
    
    **Response Example:**
    ```json
    {
      "successCount": 2,
      "failedCount": 1,
      "failedIds": [
        {
          "id": 3,
          "reason": "Cannot delay completed action"
        }
      ]
    }
    ```
    
    **Notes:**
    - Skips completed actions
    - All actions delayed by same number of days
    - Same reason appended to all notes
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        result = FollowUpService.bulk_delay_actions(
            action_ids=request.actionIds,
            delay_days=request.delayDays,
            reason=request.reason,
            user_id=user_id
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "BULK_DELAY_FAILED",
            "message": str(e),
            "message_ar": f"فشل التأجيل الجماعي: {str(e)}"
        })


# ==================== PHASE 3: BULK UPDATE ====================

@router.post("/actions/bulk-update")
async def bulk_update_actions(
    request: BulkUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update multiple actions with same values in one operation.
    
    Useful for reassigning actions or changing priorities in batch.
    
    **Request Body Example (Reassign):**
    ```json
    {
      "actionIds": [1, 2, 3],
      "assignedTo": "د. خالد محمد"
    }
    ```
    
    **Request Body Example (Change priority):**
    ```json
    {
      "actionIds": [4, 5],
      "priority": "high"
    }
    ```
    
    **Request Body Example (Move department):**
    ```json
    {
      "actionIds": [6, 7, 8],
      "departmentId": 10
    }
    ```
    
    **Response Example:**
    ```json
    {
      "successCount": 2,
      "failedCount": 1,
      "failedIds": [
        {
          "id": 3,
          "reason": "Cannot update completed action"
        }
      ]
    }
    ```
    
    **Notes:**
    - Skips completed actions
    - At least one field must be provided
    - Only provided fields are updated
    """
    require_logged_in(current_user)
    
    try:
        # Extract user ID from context
        user_id = 1  # TODO: Get from authenticated request
        
        result = FollowUpService.bulk_update_actions(
            action_ids=request.actionIds,
            assigned_to=request.assignedTo,
            priority=request.priority,
            department_id=request.departmentId,
            user_id=user_id
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_REQUEST",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "BULK_UPDATE_FAILED",
            "message": str(e),
            "message_ar": f"فشل التحديث الجماعي: {str(e)}"
        })
