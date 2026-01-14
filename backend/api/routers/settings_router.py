"""
Settings Router
FastAPI endpoints for the Settings Page.
"""

from datetime import datetime
from fastapi import APIRouter, Query, HTTPException, Body, Path
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel

from ..services.settings_service import SettingsService


router = APIRouter(prefix="/api/settings", tags=["Settings"])


# ==================== REQUEST/RESPONSE MODELS ====================

class DepartmentCreateRequest(BaseModel):
    """Request model for creating a department."""
    name: str
    name_ar: str
    code: str
    parent_id: Optional[int] = None
    mapping_mode: str = "internal"
    is_active: bool = True
    display_order: int = 0


class DepartmentUpdateRequest(BaseModel):
    """Request model for updating a department."""
    name: Optional[str] = None
    name_ar: Optional[str] = None
    code: Optional[str] = None
    parent_id: Optional[int] = None
    is_active: Optional[bool] = None
    display_order: Optional[int] = None


class AttributeValueRequest(BaseModel):
    """Request model for attribute values."""
    id: Optional[int] = None
    value: str
    value_ar: str
    code: Optional[str] = None
    color: Optional[str] = None
    display_order: int = 0
    is_active: bool = True


class AttributeUpdateRequest(BaseModel):
    """Request model for updating attributes."""
    attribute_type: str
    values: List[AttributeValueRequest]


class PolicyUpdateRequest(BaseModel):
    """Request model for policy updates."""
    policy_key: str
    policy_value: Any


class SnapshotCreateRequest(BaseModel):
    """Request model for creating snapshot."""
    snapshot_name: str
    snapshot_name_ar: str
    description: str


# ==================== B1: DEPARTMENTS - GET ====================

@router.get("/departments")
async def get_departments(
    mapping_mode: Optional[str] = Query(None, description="Filter by 'internal' or 'external'"),
    is_active: Optional[bool] = Query(True, description="Filter by active status"),
    include_children: bool = Query(True, description="Include nested children"),
    flat: bool = Query(False, description="Return flat array instead of tree")
):
    """
    Fetch all departments with optional hierarchical structure.
    
    **Parameters:**
    - `mapping_mode`: "internal" or "external" (optional)
    - `is_active`: true/false (default: true)
    - `flat`: true for flat array, false for tree (default: false)
    """
    try:
        result = SettingsService.get_departments(
            mapping_mode=mapping_mode,
            is_active=is_active if is_active is not None else True,
            include_children=include_children,
            flat=flat
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "departments_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب الأقسام: {str(e)}"
        })


# ==================== B2: DEPARTMENTS - CREATE ====================

@router.post("/departments")
async def create_department(request: DepartmentCreateRequest):
    """
    Create a new department.
    
    **Request Body:**
    ```json
    {
        "name": "Radiology",
        "name_ar": "الأشعة",
        "code": "RAD",
        "parent_id": 1,
        "mapping_mode": "internal",
        "is_active": true,
        "display_order": 5
    }
    ```
    """
    try:
        result = SettingsService.create_department(
            name=request.name,
            name_ar=request.name_ar,
            code=request.code,
            parent_id=request.parent_id,
            mapping_mode=request.mapping_mode,
            is_active=request.is_active,
            display_order=request.display_order,
            created_by_user_id=5  # TODO: Get from auth context
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail={
            "error": "duplicate_code",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "department_creation_failed",
            "message": str(e),
            "message_ar": f"فشل إنشاء القسم: {str(e)}"
        })


# ==================== B3: DEPARTMENTS - UPDATE ====================

@router.put("/departments/{department_id}")
async def update_department(
    department_id: int = Path(..., gt=0),
    request: DepartmentUpdateRequest = Body(...)
):
    """
    Update an existing department.
    
    **Path Parameter:**
    - `department_id`: Department ID to update
    """
    try:
        result = SettingsService.update_department(
            department_id=department_id,
            name=request.name,
            name_ar=request.name_ar,
            code=request.code,
            parent_id=request.parent_id,
            is_active=request.is_active,
            display_order=request.display_order,
            updated_by_user_id=5  # TODO: Get from auth context
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail={
            "error": "circular_parent_reference",
            "message": str(e),
            "message_ar": str(e)
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "department_update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث القسم: {str(e)}"
        })


# ==================== B4: DEPARTMENTS - DELETE ====================

@router.delete("/departments/{department_id}")
async def delete_department(
    department_id: int = Path(..., gt=0),
    force: bool = Query(False, description="Force delete/deactivate")
):
    """
    Delete or deactivate a department.
    
    Soft deletes department (sets is_active=false). Returns error if department
    has associated incidents unless force=true.
    """
    try:
        result = SettingsService.delete_department(
            department_id=department_id,
            force=force,
            updated_by_user_id=5  # TODO: Get from auth context
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=409, detail={
            "error": "department_has_incidents",
            "message": str(e),
            "message_ar": str(e),
            "incident_count": int(str(e).split()[7]) if "with" in str(e) else 0
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "department_deletion_failed",
            "message": str(e),
            "message_ar": f"فشل حذف القسم: {str(e)}"
        })


# ==================== B9: ATTRIBUTES - GET ====================

@router.get("/attributes")
async def get_attributes(
    attribute_type: Optional[str] = Query(None, description="Filter by specific attribute type"),
    is_active: bool = Query(True, description="Filter by active status")
):
    """
    Fetch all variable attributes.
    
    **Parameters:**
    - `attribute_type`: "severity", "domain", "category", etc. (optional)
    - `is_active`: true/false (default: true)
    """
    try:
        result = SettingsService.get_attributes(
            attribute_type=attribute_type,
            is_active=is_active
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "attributes_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب السمات: {str(e)}"
        })


# ==================== B10: ATTRIBUTES - UPDATE ====================

@router.put("/attributes")
async def update_attributes(request: AttributeUpdateRequest):
    """
    Update variable attribute values.
    
    Can add new values, update existing, or deactivate.
    
    **Request Body:**
    ```json
    {
        "attribute_type": "severity",
        "values": [
            {
                "id": 1,
                "value": "Low",
                "value_ar": "منخفض",
                "code": "LOW",
                "color": "#4caf50",
                "display_order": 1,
                "is_active": true
            }
        ]
    }
    ```
    """
    try:
        result = SettingsService.update_attributes(
            attribute_type=request.attribute_type,
            values=[v.dict() for v in request.values]
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "attribute_update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث السمات: {str(e)}"
        })


# ==================== B11: POLICIES - GET ====================

@router.get("/policies")
async def get_policies(
    category: Optional[str] = Query(None, description="Filter by category"),
    scope: Optional[str] = Query(None, description="Filter by 'global' or 'department'"),
    department_id: Optional[int] = Query(None, description="Filter by department")
):
    """
    Fetch all policy configurations.
    
    **Parameters:**
    - `category`: "reporting", "investigation", "escalation", etc. (optional)
    - `scope`: "global" or "department" (optional)
    - `department_id`: Department ID if scope=department (optional)
    """
    try:
        result = SettingsService.get_policies(
            category=category,
            scope=scope,
            department_id=department_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "policies_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب السياسات: {str(e)}"
        })


# ==================== B12: POLICIES - UPDATE ====================

@router.put("/policies")
async def update_policies(request: Dict[str, Any] = Body(...)):
    """
    Update one or more policy values.
    
    **Request Body:**
    ```json
    {
        "policies": [
            {
                "policy_key": "hcat_threshold_default",
                "policy_value": 60
            }
        ]
    }
    ```
    """
    try:
        policies = request.get('policies', [])
        result = SettingsService.update_policies(policies)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "policy_update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث السياسات: {str(e)}"
        })


# ==================== B13: EXPORT ====================

@router.get("/export")
async def export_configuration(
    format: Literal['json', 'csv'] = Query('json', description="Export format"),
    include_inactive: bool = Query(False, description="Include inactive items")
):
    """
    Export entire system configuration as JSON or CSV.
    
    **Parameters:**
    - `format`: "json" or "csv" (default: json)
    - `include_inactive`: true/false (default: false)
    """
    try:
        result = SettingsService.export_configuration(
            include_inactive=include_inactive,
            format=format
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "export_failed",
            "message": str(e),
            "message_ar": f"فشل التصدير: {str(e)}"
        })


# ==================== B15: SAVE SNAPSHOT ====================

@router.post("/save-snapshot")
async def save_snapshot(request: SnapshotCreateRequest):
    """
    Save current configuration state as a versioned snapshot for rollback.
    
    **Request Body:**
    ```json
    {
        "snapshot_name": "Pre-Migration Backup",
        "snapshot_name_ar": "نسخة احتياطية قبل الترحيل",
        "description": "Full configuration backup before system migration"
    }
    ```
    """
    try:
        result = SettingsService.save_snapshot(
            snapshot_name=request.snapshot_name,
            snapshot_name_ar=request.snapshot_name_ar,
            description=request.description,
            created_by_user_id=5  # TODO: Get from auth context
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "snapshot_save_failed",
            "message": str(e),
            "message_ar": f"فشل حفظ لقطة التكوين: {str(e)}"
        })


# ==================== SNAPSHOTS - LIST ====================

@router.get("/snapshots")
async def get_snapshots():
    """Get list of all saved configuration snapshots."""
    try:
        result = SettingsService.get_snapshots()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "snapshots_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب اللقطات: {str(e)}"
        })
