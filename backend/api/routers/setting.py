"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Feedback Configuration Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import datetime

from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/api/settings", tags=["settings"])


# -----------------------------
# Models
# -----------------------------

class FeedbackDelaySettingsResponse(BaseModel):
    delay_threshold_days: int
    last_updated: datetime
    updated_by: str


# -----------------------------
# API Endpoints
# -----------------------------

@router.get(
    "/feedback-delay",
    response_model=FeedbackDelaySettingsResponse
)
def get_feedback_delay_threshold():
    raise NotImplementedError


"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – SettingPage
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Any, Union

from fastapi import APIRouter, Query, Path, Body
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/settings", tags=["Settings"])


# =====================================================
# Shared / Utility Models
# =====================================================

class Pagination(BaseModel):
    page: int
    page_size: int
    total_records: int
    total_pages: int


# =====================================================
# A1 – Department Models
# =====================================================

class DepartmentBase(BaseModel):
    id: int
    name: str
    name_ar: str
    code: Optional[str] = None
    parent_id: Optional[int] = None
    level: Optional[int] = None
    mapping_mode: str
    is_active: bool
    display_order: Optional[int] = None


class DepartmentTreeNode(DepartmentBase):
    has_children: bool
    children: List["DepartmentTreeNode"] = []


DepartmentTreeNode.model_rebuild()


class DepartmentFlatNode(BaseModel):
    id: int
    name: str
    name_ar: str
    code: Optional[str] = None
    parent_id: Optional[int] = None
    level: int
    path: str
    depth: int


class DepartmentTreeResponse(BaseModel):
    mapping_mode: Optional[str] = None
    departments: List[DepartmentTreeNode]
    total_count: int
    active_count: int


class DepartmentFlatResponse(BaseModel):
    departments: List[DepartmentFlatNode]


class DepartmentCreateRequest(BaseModel):
    name: str
    name_ar: str
    code: Optional[str] = None
    parent_id: Optional[int] = None
    mapping_mode: str
    is_active: bool = True
    display_order: Optional[int] = None


class DepartmentCreateResponse(BaseModel):
    id: int
    name: str
    name_ar: str
    code: Optional[str] = None
    parent_id: Optional[int] = None
    level: int
    mapping_mode: str
    is_active: bool
    display_order: Optional[int] = None
    created_at: datetime
    created_by_user_id: int
    message: str
    message_ar: str


class DepartmentUpdateRequest(BaseModel):
    name: Optional[str] = None
    name_ar: Optional[str] = None
    code: Optional[str] = None
    parent_id: Optional[int] = None
    is_active: Optional[bool] = None
    display_order: Optional[int] = None


class DepartmentUpdateResponse(BaseModel):
    id: int
    name: str
    name_ar: str
    code: Optional[str] = None
    parent_id: Optional[int] = None
    level: int
    mapping_mode: str
    is_active: bool
    display_order: Optional[int] = None
    updated_at: datetime
    updated_by_user_id: int
    message: str
    message_ar: str


class DepartmentDeleteResponse(BaseModel):
    id: int
    is_active: bool
    deleted_at: datetime
    message: str
    message_ar: str


# =====================================================
# A2 – Doctor Models
# =====================================================

class DoctorBase(BaseModel):
    id: int
    employee_id: str
    name: str
    name_ar: str
    email: Optional[str] = None
    phone: Optional[str] = None
    specialty: Optional[str] = None
    specialty_ar: Optional[str] = None
    license_number: Optional[str] = None
    department_id: int
    department_name: Optional[str] = None
    department_name_ar: Optional[str] = None
    additional_departments: List[int] = []
    is_active: bool
    hire_date: Optional[date] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class DoctorListResponse(BaseModel):
    doctors: List[DoctorBase]
    pagination: Pagination


class DoctorCreateRequest(BaseModel):
    employee_id: str
    name: str
    name_ar: str
    email: Optional[str] = None
    phone: Optional[str] = None
    specialty: Optional[str] = None
    specialty_ar: Optional[str] = None
    license_number: Optional[str] = None
    department_id: int
    additional_departments: List[int] = []
    is_active: bool = True
    hire_date: Optional[date] = None


class DoctorCreateResponse(BaseModel):
    id: int
    employee_id: str
    name: str
    name_ar: str
    department_id: int
    department_name: str
    department_name_ar: str
    created_at: datetime
    message: str
    message_ar: str


class DoctorUpdateRequest(BaseModel):
    name: Optional[str] = None
    name_ar: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    specialty: Optional[str] = None
    specialty_ar: Optional[str] = None
    department_id: Optional[int] = None
    additional_departments: Optional[List[int]] = None
    is_active: Optional[bool] = None


class DoctorUpdateResponse(BaseModel):
    id: int
    employee_id: str
    name: str
    name_ar: str
    specialty: Optional[str] = None
    specialty_ar: Optional[str] = None
    department_id: int
    updated_at: datetime
    message: str
    message_ar: str


class DoctorDeleteResponse(BaseModel):
    id: int
    is_active: bool
    deleted_at: datetime
    message: str
    message_ar: str


# =====================================================
# A3 – Variable Attributes Models
# =====================================================

class AttributeValue(BaseModel):
    id: int
    value: str
    value_ar: str
    code: Optional[str] = None
    color: Optional[str] = None
    display_order: Optional[int] = None
    is_active: bool
    metadata: Optional[Dict[str, Any]] = None


class VariableAttribute(BaseModel):
    attribute_type: str
    attribute_type_label: str
    attribute_type_label_ar: str
    values: List[AttributeValue]


class VariableAttributeResponse(BaseModel):
    attributes: List[VariableAttribute]
    total_attribute_types: int
    last_updated_at: datetime


class VariableAttributeSaveRequest(BaseModel):
    attribute_type: str
    values: List[AttributeValue]


class VariableAttributeSaveResponse(BaseModel):
    attribute_type: str
    updated_count: int
    added_count: int
    deactivated_count: int
    values: List[Dict[str, Union[int, str, bool]]]
    message: str
    message_ar: str


# =====================================================
# A4 – Policy Configuration Models
# =====================================================

class Policy(BaseModel):
    policy_key: str
    policy_name: str
    policy_name_ar: str
    policy_value: Union[str, int, bool, Dict[str, Any]]
    policy_type: str
    category: str
    description: Optional[str] = None
    description_ar: Optional[str] = None
    is_global: bool
    scope: str
    department_id: Optional[int] = None
    updated_at: datetime


class PolicyListResponse(BaseModel):
    policies: List[Policy]
    total_policies: int
    global_policies: int
    department_policies: int


class PolicyUpdateItem(BaseModel):
    policy_key: str
    policy_value: Union[str, int, bool, Dict[str, Any]]


class PolicyUpdateRequest(BaseModel):
    policies: List[PolicyUpdateItem]


class PolicyUpdateResponse(BaseModel):
    updated_count: int
    policies: List[Dict[str, Union[str, int, bool, datetime]]]
    message: str
    message_ar: str


# =====================================================
# A5 – Configuration Export / Import Models
# =====================================================

class ConfigExportResponse(BaseModel):
    export_id: str
    config_version: str
    exported_at: datetime
    exported_by: str
    data: Dict[str, Any]
    metadata: Dict[str, int]


class ConfigImportRequest(BaseModel):
    import_mode: str
    overwrite_existing: bool
    data: Dict[str, Any]


class ConfigImportResponse(BaseModel):
    import_id: str
    status: str
    summary: Dict[str, int]
    errors: List[str]
    message: str
    message_ar: str


class ConfigSnapshotRequest(BaseModel):
    snapshot_name: str
    snapshot_name_ar: Optional[str] = None
    description: Optional[str] = None


class ConfigSnapshotResponse(BaseModel):
    snapshot_id: str
    snapshot_name: str
    config_version: str
    created_at: datetime
    created_by_user_id: int
    message: str
    message_ar: str


# =====================================================
# Routes – Departments
# =====================================================

@router.get(
    "/departments",
    response_model=Union[DepartmentTreeResponse, DepartmentFlatResponse],
)
def fetch_departments(
    mapping_mode: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(True),
    include_children: bool = Query(True),
    flat: bool = Query(False),
):
    """
    Fetch departments (hierarchical tree or flat list).
    """
    raise NotImplementedError


@router.post(
    "/departments",
    response_model=DepartmentCreateResponse,
    status_code=201,
)
def create_department(
    payload: DepartmentCreateRequest = Body(...),
):
    """
    Create a new department.
    """
    raise NotImplementedError


@router.put(
    "/departments/{id}",
    response_model=DepartmentUpdateResponse,
)
def update_department(
    id: int = Path(...),
    payload: DepartmentUpdateRequest = Body(...),
):
    """
    Update department details.
    """
    raise NotImplementedError


@router.delete(
    "/departments/{id}",
    response_model=DepartmentDeleteResponse,
)
def delete_department(
    id: int = Path(...),
    force: bool = Query(False),
):
    """
    Delete or deactivate a department.
    """
    raise NotImplementedError


# =====================================================
# Routes – Doctors
# =====================================================

@router.get(
    "/doctors",
    response_model=DoctorListResponse,
)
def fetch_doctors(
    department_id: Optional[int] = Query(None),
    is_active: Optional[bool] = Query(True),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """
    Fetch doctors with filtering and pagination.
    """
    raise NotImplementedError


@router.post(
    "/doctors",
    response_model=DoctorCreateResponse,
    status_code=201,
)
def create_doctor(
    payload: DoctorCreateRequest = Body(...),
):
    """
    Add a new doctor.
    """
    raise NotImplementedError


@router.put(
    "/doctors/{id}",
    response_model=DoctorUpdateResponse,
)
def update_doctor(
    id: int = Path(...),
    payload: DoctorUpdateRequest = Body(...),
):
    """
    Update doctor information.
    """
    raise NotImplementedError


@router.delete(
    "/doctors/{id}",
    response_model=DoctorDeleteResponse,
)
def delete_doctor(
    id: int = Path(...),
):
    """
    Delete or deactivate doctor.
    """
    raise NotImplementedError


# =====================================================
# Routes – Variable Attributes
# =====================================================

@router.get(
    "/attributes",
    response_model=VariableAttributeResponse,
)
def fetch_variable_attributes(
    attribute_type: Optional[str] = Query(None),
    is_active: Optional[bool] = Query(True),
):
    """
    Fetch system variable attributes.
    """
    raise NotImplementedError


@router.put(
    "/attributes",
    response_model=VariableAttributeSaveResponse,
)
def save_variable_attributes(
    payload: VariableAttributeSaveRequest = Body(...),
):
    """
    Save variable attribute values.
    """
    raise NotImplementedError


# =====================================================
# Routes – Policies
# =====================================================

@router.get(
    "/policies",
    response_model=PolicyListResponse,
)
def fetch_policies(
    category: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
    department_id: Optional[int] = Query(None),
):
    """
    Fetch policy configuration.
    """
    raise NotImplementedError


@router.put(
    "/policies",
    response_model=PolicyUpdateResponse,
)
def update_policies(
    payload: PolicyUpdateRequest = Body(...),
):
    """
    Update policy configuration values.
    """
    raise NotImplementedError


# =====================================================
# Routes – Configuration Export / Import / Snapshot
# =====================================================

@router.get(
    "/export",
    response_model=ConfigExportResponse,
)
def export_configuration(
    format: str = Query("json"),
    include_inactive: bool = Query(False),
):
    """
    Export full system configuration.
    """
    raise NotImplementedError


@router.post(
    "/import",
    response_model=ConfigImportResponse,
)
def import_configuration(
    payload: ConfigImportRequest = Body(...),
):
    """
    Import system configuration.
    """
    raise NotImplementedError


@router.post(
    "/save-snapshot",
    response_model=ConfigSnapshotResponse,
    status_code=201,
)
def save_configuration_snapshot(
    payload: ConfigSnapshotRequest = Body(...),
):
    """
    Save configuration snapshot.
    """
    raise NotImplementedError
