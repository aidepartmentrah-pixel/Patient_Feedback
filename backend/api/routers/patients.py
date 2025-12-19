"""
AUTO-GENERATED API CONTRACT

Source: Obsidian – Patient History Page
Iteration: 1
Status: API skeleton only – no implementation
"""

from datetime import date, datetime
from typing import Optional, List, Dict, Union

from fastapi import APIRouter, Query, Path
from pydantic import BaseModel

router = APIRouter(prefix="/api/patients", tags=["Patients"])


# =====================================================
# Search Models
# =====================================================

class PatientSearchItem(BaseModel):
    patient_id: Union[int, str]
    mrn: str
    full_name: str
    date_of_birth: date
    age: Optional[int] = None
    gender: str
    phone: Optional[str] = None


class PatientSearchResponse(BaseModel):
    patients: List[PatientSearchItem]
    total: int


# =====================================================
# Patient Profile Models
# =====================================================

class PatientProfile(BaseModel):
    patient_id: Union[int, str]
    mrn: str
    full_name: str
    full_name_en: Optional[str] = None
    date_of_birth: date
    age: Optional[int] = None
    gender: str
    nationality: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    emergency_contact: Optional[str] = None
    emergency_phone: Optional[str] = None
    total_incidents: int
    last_visit_date: Optional[date] = None
    registration_date: Optional[date] = None


# =====================================================
# Incident List Models
# =====================================================

class PatientIncidentItem(BaseModel):
    incident_id: Union[int, str]
    record_id: str
    date: date
    feedback_received_date: date
    department: str
    department_ar: str
    category: str
    category_ar: str
    severity: str
    doctor_name: Optional[str] = None
    status: str
    description: Optional[str] = None
    is_red_flag: bool
    is_never_event: bool


class PatientIncidentListResponse(BaseModel):
    patient_id: Union[int, str]
    patient_name: str
    incidents: List[PatientIncidentItem]
    total: int
    limit: int
    offset: int


# =====================================================
# Incident Detail Models
# =====================================================

class PatientIncidentDetails(BaseModel):
    incident_id: Union[int, str]
    record_id: str
    date: date
    feedback_received_date: date
    patient_id: Union[int, str]
    patient_name: str
    department: str
    target_department: Optional[str] = None
    category: str
    category_ar: str
    classification: Optional[str] = None
    severity: str
    harm_level: Optional[str] = None
    stage: Optional[str] = None
    doctor_name: Optional[str] = None
    status: str
    complaint_text: str
    immediate_action: Optional[str] = None
    taken_action: Optional[str] = None
    is_red_flag: bool
    is_never_event: bool
    created_at: datetime
    last_updated_at: datetime


# =====================================================
# Combined Full History Model
# =====================================================

class PatientFullHistoryResponse(BaseModel):
    profile: PatientProfile
    incidents: PatientIncidentListResponse


# =====================================================
# Routes
# =====================================================

@router.get(
    "/search",
    response_model=PatientSearchResponse,
)
def search_patients(
    query: Optional[str] = Query(None),
    mrn: Optional[str] = Query(None),
    phone: Optional[str] = Query(None),
    date_of_birth: Optional[date] = Query(None),
    limit: int = Query(50, ge=1),
):
    """
    Search patients by name, MRN, phone, or DOB.
    """
    raise NotImplementedError


@router.get(
    "/{patient_id}/profile",
    response_model=PatientProfile,
)
def get_patient_profile(
    patient_id: Union[int, str] = Path(...),
):
    """
    Fetch detailed patient profile.
    """
    raise NotImplementedError


@router.get(
    "/{patient_id}/incidents",
    response_model=PatientIncidentListResponse,
)
def get_patient_incidents(
    patient_id: Union[int, str] = Path(...),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    department: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
):
    """
    Fetch all incidents related to a patient.
    """
    raise NotImplementedError


@router.get(
    "/{patient_id}/incidents/{incident_id}",
    response_model=PatientIncidentDetails,
)
def get_patient_incident_details(
    patient_id: Union[int, str] = Path(...),
    incident_id: Union[int, str] = Path(...),
):
    """
    Fetch full details of a specific patient incident.
    """
    raise NotImplementedError


@router.get(
    "/{patient_id}/full-history",
    response_model=PatientFullHistoryResponse,
)
def get_patient_full_history(
    patient_id: Union[int, str] = Path(...),
    from_date: Optional[date] = Query(None),
    to_date: Optional[date] = Query(None),
    department: Optional[str] = Query(None),
    severity: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
):
    """
    Fetch patient profile and incidents in a single request.
    """
    raise NotImplementedError
