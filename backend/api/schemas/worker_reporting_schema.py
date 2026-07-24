"""
Worker Reporting Schema
Pydantic models for worker profile and reporting endpoints.

This module defines request and response schemas for worker (employee) profile
and reporting functionality. Worker data is sourced from HR employee views and
aggregated from incident and action item tables.

Note:
    - Metrics are computed by the service layer, not here.
    - Identity information comes from APP_VIEWTABLE_HR_EMPLOYEES.
    - This schema defines contracts only, no business logic.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Union
from datetime import date


class WorkerIdentityBlock(BaseModel):
    """
    Worker identity information from HR system.
    
    This block contains basic employee identification and organizational
    assignment information. All fields are sourced from the HR employee view.
    
    Attributes:
        employee_id: Unique employee identifier from HR system
        full_name: Employee's full name
        job_title: Current job title or position (may be null)
        department_id: Assigned department ID (may be null)
        section_id: Assigned section ID within department (may be null)
        administration_id: Assigned administration ID (may be null)
        is_active: Whether the employee is currently active
    """
    # Union[int, str]: a plain reserve EmployeeID (int), or an opaque
    # external id (str) for a worker sourced from the Hospital Directory API
    # who has never appeared in a local incident -- their profile is a
    # valid, honest "zero history" result, not an error. See
    # WorkerReportingService.get_worker_profile's early-return branch.
    employee_id: Union[int, str] = Field(..., description="Unique employee identifier (int for reserve, opaque string for external/unmaterialized)")
    full_name: str = Field(..., description="Employee's full name")
    job_title: Optional[str] = Field(None, description="Current job title or position")
    department_id: Optional[int] = Field(None, description="Assigned department ID")
    section_id: Optional[int] = Field(None, description="Assigned section ID")
    administration_id: Optional[int] = Field(None, description="Assigned administration ID")
    is_active: Optional[bool] = Field(None, description="Whether employee is active")

    class Config:
        json_schema_extra = {
            "example": {
                "employee_id": 12345,
                "full_name": "Ahmed Mohammed Al-Shahrani",
                "job_title": "Quality Assurance Specialist",
                "department_id": 42,
                "section_id": 8,
                "administration_id": 3,
                "is_active": True
            }
        }


class WorkerMetricBlock(BaseModel):
    """
    Aggregated performance metrics for a worker.
    
    This block contains computed metrics derived from incident cases,
    action items, and explanation workflow data. All metrics are calculated
    by the service layer based on the specified time period.
    
    Attributes:
        total_incidents: Total number of incidents involving this worker
        total_action_items: Total action items assigned to this worker
        completed_action_items: Number of completed action items
        overdue_action_items: Number of overdue action items
        explanation_rejected_count: Number of rejected explanations
        explanation_accepted_count: Number of accepted explanations
    
    Note:
        All counts default to 0 if no data exists for the specified period.
    """
    total_incidents: int = Field(0, ge=0, description="Total incidents involving worker")
    high_severity: int = Field(0, ge=0, description="High severity incidents")
    medium_severity: int = Field(0, ge=0, description="Medium severity incidents")
    low_severity: int = Field(0, ge=0, description="Low severity incidents")
    good_feedback_count: int = Field(0, ge=0, description="Good feedback (Notice/تنويه)")
    bad_feedback_count: int = Field(0, ge=0, description="Bad feedback (Critique/نقد)")
    neutral_feedback_count: int = Field(0, ge=0, description="Neutral feedback")
    total_action_items: int = Field(0, ge=0, description="Total assigned action items")
    completed_action_items: int = Field(0, ge=0, description="Completed action items")
    overdue_action_items: int = Field(0, ge=0, description="Overdue action items")
    explanation_rejected_count: int = Field(0, ge=0, description="Rejected explanations")
    explanation_accepted_count: int = Field(0, ge=0, description="Accepted explanations")

    class Config:
        json_schema_extra = {
            "example": {
                "total_incidents": 12,
                "high_severity": 2,
                "medium_severity": 4,
                "low_severity": 6,
                "good_feedback_count": 3,
                "bad_feedback_count": 7,
                "neutral_feedback_count": 2,
                "total_action_items": 45,
                "completed_action_items": 38,
                "overdue_action_items": 3,
                "explanation_rejected_count": 2,
                "explanation_accepted_count": 15
            }
        }


class WorkerProfileResponse(BaseModel):
    """
    Complete worker profile response including identity and metrics.
    
    This response model combines worker identity information with aggregated
    performance metrics for a specified time period. Used by worker profile
    endpoints to provide comprehensive worker information.
    
    Attributes:
        worker: Worker identity block from HR system
        metrics: Aggregated performance metrics block
        period_from: Start date of metrics aggregation period (optional)
        period_to: End date of metrics aggregation period (optional)
    
    Note:
        If period_from/period_to are None, metrics represent all-time aggregation.
    """
    worker: WorkerIdentityBlock = Field(..., description="Worker identity information")
    metrics: WorkerMetricBlock = Field(..., description="Aggregated performance metrics")
    incidents: List[dict] = Field(default_factory=list, description="Detailed incidents list")
    period_from: Optional[date] = Field(None, description="Metrics period start date")
    period_to: Optional[date] = Field(None, description="Metrics period end date")

    class Config:
        json_schema_extra = {
            "example": {
                "worker": {
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
                "period_from": "2025-01-01",
                "period_to": "2025-12-31"
            }
        }
