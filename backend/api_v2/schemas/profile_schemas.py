"""
V2 Profile Response Schemas
Phase B — B-B5 — Contract consistency for V2 profile payloads.

This module defines standardized response schemas for all V2 profile endpoints:
- /api/v2/doctors/{id}/profile or full-report
- /api/v2/patients/{id}/profile or full-history
- /api/v2/workers/{id}/profile

All profile endpoints share the same top-level structure for frontend consistency.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import date, datetime


# ============================================================
# COMMON TOP-LEVEL STRUCTURE (Phase B — B-B5)
# ============================================================

class EntityMeta(BaseModel):
    """
    Metadata block for entity profile responses.
    
    Attributes:
        entity_type: Type of entity ('doctor', 'patient', 'worker')
        entity_id: Unique identifier for the entity
        period_from: Optional start date for metrics/data filtering
        period_to: Optional end date for metrics/data filtering
    """
    entity_type: str = Field(..., description="Entity type: doctor, patient, or worker")
    entity_id: Union[int, str] = Field(
        ...,
        description="Entity unique identifier — int for doctors/workers/reserve patients, "
                    "opaque string for Hospital Directory API (external) patients",
    )
    period_from: Optional[date] = Field(None, description="Start date for metrics period")
    period_to: Optional[date] = Field(None, description="End date for metrics period")

    class Config:
        json_schema_extra = {
            "example": {
                "entity_type": "doctor",
                "entity_id": 12345,
                "period_from": "2025-01-01",
                "period_to": "2025-12-31"
            }
        }


# ============================================================
# DOCTOR PROFILE V2 SCHEMA
# ============================================================

class DoctorProfileV2Response(BaseModel):
    """
    Standardized V2 response for doctor profile endpoints.
    
    Phase B — V2 profile contract normalized.
    
    Attributes:
        profile: Doctor identity and basic information
        metrics: Aggregated performance metrics
        items: Related incidents or actions (empty array if not applicable)
        meta: Entity metadata
    """
    profile: Dict[str, Any] = Field(..., description="Doctor identity and basic information")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Related incidents or actions")
    meta: EntityMeta = Field(..., description="Entity metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "profile": {
                    "id": 12345,
                    "name_en": "Dr. Ahmed Al-Shahrani",
                    "name_ar": "د. أحمد الشهراني",
                    "specialty": "Cardiology",
                    "status": "active",
                    "source": "hospital",
                    "source_system": "HIS"
                },
                "metrics": {
                    "total_incidents": 45,
                    "incident_severity_breakdown": {
                        "Minor": 20,
                        "Major": 15,
                        "Event": 10
                    }
                },
                "items": [],
                "meta": {
                    "entity_type": "doctor",
                    "entity_id": 12345,
                    "period_from": "2025-01-01",
                    "period_to": "2025-12-31"
                }
            }
        }


# ============================================================
# PATIENT PROFILE V2 SCHEMA
# ============================================================

class PatientProfileV2Response(BaseModel):
    """
    Standardized V2 response for patient profile endpoints.
    
    Phase B — V2 profile contract normalized.
    
    Attributes:
        profile: Patient identity and demographics
        metrics: Aggregated patient metrics (visit count, etc.)
        items: Patient incidents/history (empty array if not applicable)
        meta: Entity metadata
    """
    profile: Dict[str, Any] = Field(..., description="Patient identity and demographics")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Patient metrics")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Patient incidents or history")
    meta: EntityMeta = Field(..., description="Entity metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "profile": {
                    "PatientID": 789,
                    "MRN": "MRN123456",
                    "PatientName": "محمد علي",
                    "PatientNameEnglish": "Mohammed Ali",
                    "DateOfBirth": "1980-05-15",
                    "Age": 44,
                    "Gender": "Male",
                    "Phone": "+966501234567"
                },
                "metrics": {
                    "TotalIncidents": 3,
                    "LastVisitDate": "2025-12-15"
                },
                "items": [],
                "meta": {
                    "entity_type": "patient",
                    "entity_id": 789,
                    "period_from": None,
                    "period_to": None
                }
            }
        }


# ============================================================
# WORKER PROFILE V2 SCHEMA
# ============================================================

class WorkerProfileV2Response(BaseModel):
    """
    Standardized V2 response for worker profile endpoints.
    
    Phase B — V2 profile contract normalized.
    
    Attributes:
        profile: Worker identity and organizational assignment
        metrics: Aggregated performance metrics
        items: Related action items or incidents (empty array if not applicable)
        meta: Entity metadata
    """
    profile: Dict[str, Any] = Field(..., description="Worker identity and org assignment")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    items: List[Dict[str, Any]] = Field(default_factory=list, description="Action items or incidents")
    meta: EntityMeta = Field(..., description="Entity metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "profile": {
                    "employee_id": 12345,
                    "full_name": "Ahmed Mohammed Al-Shahrani",
                    "job_title": "Quality Assurance Specialist",
                    "department_id": 42,
                    "section_id": 8,
                    "administration_id": 3,
                    "is_active": True
                },
                "metrics": {
                    "total_incidents": 12,
                    "total_action_items": 45,
                    "completed_action_items": 38,
                    "overdue_action_items": 3,
                    "explanation_rejected_count": 2,
                    "explanation_accepted_count": 15
                },
                "items": [],
                "meta": {
                    "entity_type": "worker",
                    "entity_id": 12345,
                    "period_from": "2025-01-01",
                    "period_to": "2025-12-31"
                }
            }
        }
