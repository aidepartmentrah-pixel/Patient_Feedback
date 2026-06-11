"""
Settings Router
FastAPI endpoints for the Settings Page.
"""

from datetime import datetime
from fastapi import APIRouter, Query, HTTPException, Body, Path, Depends
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in, require_software_admin
from ..services.settings_service import SettingsService
from ..services.doctors_service import DoctorService


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
    flat: bool = Query(False, description="Return flat array instead of tree"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Fetch all departments with optional hierarchical structure.
    
    **Parameters:**
    - `mapping_mode`: "internal" or "external" (optional)
    - `is_active`: true/false (default: true)
    - `flat`: true for flat array, false for tree (default: false)
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
async def create_department(
    request: DepartmentCreateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
    request: DepartmentUpdateRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update an existing department.
    
    **Path Parameter:**
    - `department_id`: Department ID to update
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
    force: bool = Query(False, description="Force delete/deactivate"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Delete or deactivate a department.
    
    Soft deletes department (sets is_active=false). Returns error if department
    has associated incidents unless force=true.
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
    is_active: bool = Query(True, description="Filter by active status"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Fetch all variable attributes.
    
    **Parameters:**
    - `attribute_type`: "severity", "domain", "category", etc. (optional)
    - `is_active`: true/false (default: true)
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
async def update_attributes(
    request: AttributeUpdateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
    department_id: Optional[int] = Query(None, description="Filter by department"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Fetch all policy configurations.
    
    **Parameters:**
    - `category`: "reporting", "investigation", "escalation", etc. (optional)
    - `scope`: "global" or "department" (optional)
    - `department_id`: Department ID if scope=department (optional)
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
async def update_policies(
    request: Dict[str, Any] = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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


# ==================== B12a: POLICY - GET BY DEPARTMENT ====================

@router.get("/policy/{department_id}")
async def get_policy_by_department(
    department_id: int = Path(..., ge=1, description="Department ID"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Fetch policy configuration for a specific department.
    
    Returns combined global policies and department-specific overrides.
    
    **Path Parameters:**
    - `department_id`: The department ID to fetch policy for
    
    **Response:**
    ```json
    {
      "success": true,
      "department_id": 20,
      "policy": {
        "slaThresholds": { "critical": 24, "high": 48, ... },
        "notificationRules": { ... },
        ...
      }
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.get_department_policy(department_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "policy_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب السياسة: {str(e)}"
        })


# ==================== B12b: POLICY - SAVE BY DEPARTMENT ====================

@router.put("/policy/{department_id}")
async def save_policy_by_department(
    department_id: int = Path(..., ge=1, description="Department ID"),
    request: Dict[str, Any] = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Save policy configuration for a specific department.
    
    **Path Parameters:**
    - `department_id`: The department ID to save policy for
    
    **Request Body:**
    ```json
    {
      "slaThresholds": { "critical": 24, "high": 48, ... },
      "notificationRules": { ... },
      "escalationConfig": { ... },
      ...
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.save_department_policy(department_id, request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "policy_save_failed",
            "message": str(e),
            "message_ar": f"فشل حفظ السياسة: {str(e)}"
        })


# ==================== POLICY COVERAGE VALIDATION ====================

@router.get("/policy/validation")
async def validate_policy_coverage(
    unit_type: Optional[int] = Query(
        None,
        description="Restrict to one org-unit type code: 323=Administration, 325=Department, 324=Section. Omit for all."
    ),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Scan every active organizational unit and report its policy assignment status.

    **Status values:**
    - `ok`               – active policy with at least one compliance rule enabled
    - `no_rules_enabled` – policy record exists and is active, but ALL Enable* flags are 0
    - `inactive`         – policy record exists but IsActive = 0
    - `missing`          – no row at all in APP_OrgUnitPolicy

    **Example response:**
    ```json
    {
      "summary": { "total": 15, "ok": 10, "missing": 3, "inactive": 1, "no_rules_enabled": 1 },
      "units": [
        { "id": 5, "name": "إدارة الجودة", "type_label": "إدارة", "status": "missing",
          "issues": ["لا يوجد سجل سياسة (No policy record in APP_OrgUnitPolicy)"] },
        ...
      ]
    }
    ```
    """
    require_logged_in(current_user)

    try:
        from ..db_layer.org_unit_policy import validate_policy_coverage as _validate
        result = _validate(unit_type_filter=unit_type)
        return {"success": True, **result}
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "policy_validation_failed",
            "message": str(e),
            "message_ar": f"فشل التحقق من سياسات الوحدات: {str(e)}"
        })


# ==================== B13: EXPORT ====================

@router.get("/export")
async def export_configuration(
    format: Literal['json', 'csv'] = Query('json', description="Export format"),
    include_inactive: bool = Query(False, description="Include inactive items"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Export entire system configuration as JSON or CSV.
    
    **Parameters:**
    - `format`: "json" or "csv" (default: json)
    - `include_inactive`: true/false (default: false)
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
async def save_snapshot(
    request: SnapshotCreateRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
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
    require_logged_in(current_user)
    require_software_admin(current_user)
    
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
async def get_snapshots(current_user: CurrentUser = Depends(get_current_user)):
    """Get list of all saved configuration snapshots."""
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.get_snapshots()
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "snapshots_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب اللقطات: {str(e)}"
        })


# ==================== SYSTEM SETTINGS ====================

class SystemSettingRequest(BaseModel):
    """Request model for creating/updating system settings."""
    setting_key: str
    setting_value: str
    label: Optional[str] = None
    label_ar: Optional[str] = None
    setting_type: str = "text"  # text, number, boolean, json
    description: Optional[str] = None
    description_ar: Optional[str] = None


class SystemSettingUpdateRequest(BaseModel):
    """Request model for updating a setting value only."""
    setting_value: str


@router.get("/system-settings")
async def get_system_settings(
    is_active: bool = Query(True, description="Filter by active status"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get all system settings.
    
    **Query Parameters:**
    - `is_active`: true/false (default: true)
    
    **Example Response:**
    ```json
    {
      "settings": [
        {
          "id": 1,
          "setting_key": "max_file_size_mb",
          "setting_value": "10",
          "label": "Maximum File Size (MB)",
          "label_ar": "الحد الأقصى لحجم الملف",
          "setting_type": "number",
          "description": "Maximum upload file size in megabytes",
          "description_ar": "الحد الأقصى لحجم رفع الملف بالميغابايت",
          "is_active": true,
          "created_at": "2026-01-21 10:00:00",
          "updated_at": "2026-01-21 10:00:00"
        }
      ],
      "total": 4
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.get_system_settings(is_active=is_active)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "settings_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب الإعدادات: {str(e)}"
        })


@router.get("/system-settings/{setting_key}")
async def get_system_setting(
    setting_key: str = Path(..., description="Setting key to retrieve"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Get a specific system setting by key.
    
    **Path Parameters:**
    - `setting_key`: The unique key of the setting (e.g., "max_file_size_mb")
    
    **Example Response:**
    ```json
    {
      "id": 1,
      "setting_key": "max_file_size_mb",
      "setting_value": "10",
      "label": "Maximum File Size (MB)",
      "label_ar": "الحد الأقصى لحجم الملف",
      "setting_type": "number",
      "is_active": true
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.get_setting(setting_key)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail={
            "error": "setting_not_found",
            "message": str(ve),
            "message_ar": "الإعداد غير موجود"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "setting_fetch_failed",
            "message": str(e),
            "message_ar": f"فشل جلب الإعداد: {str(e)}"
        })


@router.put("/system-settings/{setting_key}")
async def update_system_setting(
    setting_key: str = Path(..., description="Setting key to update"),
    request: SystemSettingUpdateRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Update a system setting value.
    
    **Path Parameters:**
    - `setting_key`: The unique key of the setting
    
    **Request Body:**
    ```json
    {
      "setting_value": "20"
    }
    ```
    
    **Example Response:**
    ```json
    {
      "success": true,
      "setting": {
        "id": 1,
        "setting_key": "max_file_size_mb",
        "setting_value": "20",
        "label": "Maximum File Size (MB)",
        "updated_at": "2026-01-21 11:30:00"
      },
      "message": "Setting 'max_file_size_mb' updated successfully",
      "message_ar": "تم تحديث الإعداد 'max_file_size_mb' بنجاح"
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.update_setting(
            setting_key=setting_key,
            setting_value=request.setting_value,
            updated_by=None  # TODO: Get from auth context
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail={
            "error": "setting_not_found",
            "message": str(ve),
            "message_ar": "الإعداد غير موجود"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "setting_update_failed",
            "message": str(e),
            "message_ar": f"فشل تحديث الإعداد: {str(e)}"
        })


@router.post("/system-settings")
async def create_system_setting(
    request: SystemSettingRequest,
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Create a new system setting.
    
    **Request Body:**
    ```json
    {
      "setting_key": "new_parameter_name",
      "setting_value": "default_value",
      "label": "My New Parameter",
      "label_ar": "المتغير الجديد",
      "setting_type": "text",
      "description": "Description of the parameter",
      "description_ar": "وصف المتغير"
    }
    ```
    
    **Example Response:**
    ```json
    {
      "success": true,
      "setting": {
        "id": 5,
        "setting_key": "new_parameter_name",
        "setting_value": "default_value",
        "label": "My New Parameter"
      },
      "message": "Setting 'new_parameter_name' created successfully",
      "message_ar": "تم إنشاء الإعداد 'new_parameter_name' بنجاح"
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = SettingsService.create_setting(
            setting_key=request.setting_key,
            setting_value=request.setting_value,
            label=request.label,
            label_ar=request.label_ar,
            setting_type=request.setting_type,
            description=request.description,
            description_ar=request.description_ar,
            created_by=None  # TODO: Get from auth context
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail={
            "error": "validation_error",
            "message": str(ve),
            "message_ar": "خطأ في التحقق من الصحة"
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail={
            "error": "setting_create_failed",
            "message": str(e),
            "message_ar": f"فشل إنشاء الإعداد: {str(e)}"
        })


# ==================== B8: DOCTORS - DELETE ====================

@router.delete("/doctors/{doctor_id}")
async def delete_doctor(
    doctor_id: int = Path(..., description="Doctor ID to delete"),
    force: bool = Query(False, description="If true, permanently delete (only if no incidents)"),
    current_user: CurrentUser = Depends(get_current_user)
):
    """
    Delete or deactivate a doctor from the reserve table (APP_RESERVE_DOCTOR).
    
    **IMPORTANT**: This endpoint ONLY operates on user-created doctors in the 
    reserve table. Hospital doctors (from VW_Doctors/APP_LOOKUP_DOCTOR) are 
    read-only and cannot be deleted.
    
    **Behavior:**
    - If doctor has associated incidents: Doctor is deactivated (soft-delete)
    - If doctor has no incidents and force=false: Doctor is deactivated
    - If doctor has no incidents and force=true: Doctor is permanently deleted
    
    **Response** (200 OK - Soft Delete):
    ```json
    {
      "success": true,
      "id": 157,
      "is_active": false,
      "deleted_at": null,
      "action": "deactivated",
      "incident_count": 45,
      "message": "Doctor deactivated successfully",
      "message_ar": "تم إلغاء تنشيط الطبيب بنجاح"
    }
    ```
    
    **Response** (404 Not Found):
    ```json
    {
      "error": "doctor_not_found",
      "message": "Doctor with ID 999 not found in reserve table"
    }
    ```
    
    **Response** (403 Forbidden - Hospital Doctor):
    ```json
    {
      "error": "hospital_doctor_readonly",
      "message": "Doctor ID 123 is from the hospital system and cannot be deleted."
    }
    ```
    """
    require_logged_in(current_user)
    require_software_admin(current_user)
    
    try:
        result = DoctorService.delete_doctor(
            doctor_id=doctor_id,
            force=force
        )
        return result
    except ValueError as ve:
        error_msg = str(ve)
        if "hospital system" in error_msg.lower():
            raise HTTPException(status_code=403, detail={
                "error": "hospital_doctor_readonly",
                "message": error_msg,
                "message_ar": "لا يمكن حذف طبيب من نظام المستشفى. الأطباء في نظام المستشفى للقراءة فقط."
            })
        raise HTTPException(status_code=404, detail={
            "error": "doctor_not_found",
            "message": error_msg,
            "message_ar": "الطبيب غير موجود في جدول الاحتياط"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "doctor_delete_failed",
            "message": str(e),
            "message_ar": f"فشل حذف الطبيب: {str(e)}"
        })


# ==================== SECTION MANAGEMENT ====================

class SectionRenameRequest(BaseModel):
    name: str
    name_ar: Optional[str] = None
    parent_id: Optional[int] = None  # reassign to a different department


@router.get("/sections")
async def list_sections(
    current_user: CurrentUser = Depends(get_current_user),
):
    """List all sections (leaves) with their parent department info."""
    require_logged_in(current_user)
    try:
        from core.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                s.UniqueID AS section_id,
                s.Name AS section_name,
                s.ParentID AS department_id,
                d.Name AS department_name,
                d.ParentID AS administration_id,
                a.Name AS administration_name
            FROM AdminsrationUnit s
            LEFT JOIN AdminsrationUnit d ON s.ParentID = d.UniqueID
            LEFT JOIN AdminsrationUnit a ON d.ParentID = a.UniqueID
            WHERE s.Type = 324 AND s.Frozen = 0
            ORDER BY a.Name, d.Name, s.Name
            """
        )
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description]
        conn.close()
        return {"sections": [dict(zip(cols, r)) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "fetch_failed", "message": str(e)})


@router.get("/sections/departments")
async def list_departments_for_reassignment(
    current_user: CurrentUser = Depends(get_current_user),
):
    """Return all departments (Type=2) that sections can be reassigned to."""
    require_logged_in(current_user)
    try:
        from core.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                d.UniqueID AS department_id,
                d.Name AS department_name,
                d.ParentID AS administration_id,
                a.Name AS administration_name
            FROM AdminsrationUnit d
            LEFT JOIN AdminsrationUnit a ON d.ParentID = a.UniqueID
            WHERE d.Type IN (323, 325) AND d.Frozen = 0
            ORDER BY a.Name, d.Name
            """
        )
        rows = cursor.fetchall()
        cols = [c[0] for c in cursor.description]
        conn.close()
        return {"departments": [dict(zip(cols, r)) for r in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "fetch_failed", "message": str(e)})


@router.put("/sections/{section_id}")
async def update_section(
    section_id: int = Path(..., gt=0),
    request: SectionRenameRequest = Body(...),
    current_user: CurrentUser = Depends(get_current_user),
):
    """
    Rename a section and/or reassign it to a different department.
    Only changes Name and/or ParentID — no historical data is touched.
    """
    require_logged_in(current_user)
    try:
        from core.database import get_connection
        conn = get_connection()
        cursor = conn.cursor()

        # Verify section exists and is Type=3
        cursor.execute(
            "SELECT UniqueID, Name, ParentID, Type FROM AdminsrationUnit WHERE UniqueID = ?",
            section_id,
        )
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail={"error": "NOT_FOUND", "message": f"Section {section_id} not found"})
        if row.Type != 324:
            conn.close()
            raise HTTPException(status_code=400, detail={"error": "NOT_SECTION", "message": "Unit is not a section (Type must be 324)"})

        new_name = request.name.strip()
        new_parent = request.parent_id if request.parent_id is not None else row.ParentID

        if request.parent_id is not None:
            # Verify the target department is Type=2
            cursor.execute("SELECT Type FROM AdminsrationUnit WHERE UniqueID = ?", new_parent)
            dept_row = cursor.fetchone()
            if not dept_row or dept_row.Type not in (323, 325):
                conn.close()
                raise HTTPException(status_code=400, detail={"error": "INVALID_DEPARTMENT", "message": "Target parent must be a department (Type=323 or 325)"})

        cursor.execute(
            "UPDATE AdminsrationUnit SET Name = ?, ParentID = ? WHERE UniqueID = ?",
            new_name, new_parent, section_id,
        )
        conn.commit()
        conn.close()
        return {
            "success": True,
            "section_id": section_id,
            "name": new_name,
            "department_id": new_parent,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "UPDATE_FAILED", "message": str(e)})
