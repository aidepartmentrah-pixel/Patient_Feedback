from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class CaseAction:
    immediate_action: Optional[str] = None           # maps to IncidentRequestCaseAction.Description
    taken_action: Optional[str] = None               # maps to IncidentRequestCaseAction.SectionNote / DepartmentNote
    improvement_opportunity_type: Optional[bool] = None  # maps to IncidentRequestCaseAction.IsImprovementForm

@dataclass
class IncidentCase:
    domain: Optional[str] = None                     # maps to IncidentRequestCase.IncidentCaseCategoryID
    category: Optional[str] = None                   # maps to IncidentRequestCase.IncidentCaseSubCategoryID
    sub_category: Optional[str] = None               # maps to IncidentRequestCase.IncidentCaseSubCategoryID
    classification_ar: Optional[str] = None          # new column
    classification_en: Optional[str] = None          # new column
    complaint_text: Optional[str] = None             # maps to IncidentRequestCase.Description
    severity_level: Optional[int] = None             # maps to Parameter.Severity
    stage: Optional[str] = None                      # new column
    harm_level: Optional[str] = None                 # new column
    status: Optional[int] = None                     # maps to IncidentRequestCase.IncidentRequestCaseStatusID
    target_department: Optional[int] = None          # maps to IncidentRequestCase.SectionID / AdminID
    actions: List[CaseAction] = field(default_factory=list)

@dataclass
class UniversalIncidentRecord:
    unique_id: Optional[int] = None  # DB primary key
    feedback_received_date: Optional[datetime] = None    # IncidentRequest.DateAndTimeRecieved
    record_id: Optional[str] = None                      # IncidentRequest.Code or UniqueID
    patient_full_name: Optional[str] = None              # IncidentRequest.PatientName
    issuing_department: Optional[int] = None             # IncidentRequest.SourceSectionID
    source_1: Optional[int] = None                        # IncidentRequest.IncidentSourceID
    feedback_type: Optional[int] = None                  # IncidentRequest.IncidentRequesterTypeID
    cases: List[IncidentCase] = field(default_factory=list)
