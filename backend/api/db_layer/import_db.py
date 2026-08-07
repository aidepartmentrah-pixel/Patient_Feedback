"""
Import DB Layer — Hospital Data Intake Pipeline
Pure SQL operations only. No business logic.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, date
from core.database import get_connection


# ============================================================
# LOOKUP LOADING
# ============================================================

def load_all_lookups() -> Dict[str, Any]:
    """
    Load every lookup table into memory for validation.
    Returns two structures per category:
      - maps:  {lowercase_name: id}  — for validation
      - lists: [name, ...]           — for template dropdowns (ordered)
    Also returns classification_chains: {classification_id: {domain_id, category_id, subcategory_id}}
    """
    conn = get_connection()
    cursor = conn.cursor()
    maps = {}
    lists = {}

    try:
        # Feedback Intent Types
        cursor.execute("SELECT FeedbackIntentTypeID, NameAr FROM dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE WHERE IsActive=1 ORDER BY DisplayOrder")
        rows = cursor.fetchall()
        maps["feedback_types"] = {(r.NameAr or "").lower().strip(): r.FeedbackIntentTypeID for r in rows if r.NameAr}
        lists["feedback_types"] = [r.NameAr for r in rows if r.NameAr]

        # Sources
        cursor.execute("SELECT SourceID, SourceNameAr FROM dbo.APP_LOOKUP_SOURCE WHERE IsActive=1 ORDER BY DisplayOrder")
        rows = cursor.fetchall()
        maps["sources"] = {(r.SourceNameAr or "").lower().strip(): r.SourceID for r in rows if r.SourceNameAr}
        lists["sources"] = [r.SourceNameAr for r in rows if r.SourceNameAr]

        # Org Units (shared for issuing dept and target dept)
        cursor.execute("SELECT UniqueID, Name FROM dbo.AdminsrationUnit WHERE Frozen=0 ORDER BY Name")
        rows = cursor.fetchall()
        maps["org_units"] = {(r.Name or "").lower().strip(): r.UniqueID for r in rows if r.Name}
        lists["org_units"] = [r.Name for r in rows if r.Name]

        # Domains
        cursor.execute("SELECT DomainID, DomainName FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainOrder")
        rows = cursor.fetchall()
        maps["domains"] = {(r.DomainName or "").lower().strip(): r.DomainID for r in rows if r.DomainName}
        maps["domain_names"] = {r.DomainID: r.DomainName for r in rows if r.DomainName}
        lists["domains"] = [r.DomainName for r in rows if r.DomainName]

        # Categories
        cursor.execute("SELECT CategoryID, CategoryName FROM dbo.APP_LOOKUP_CATEGORY ORDER BY CategoryOrder")
        rows = cursor.fetchall()
        maps["categories"] = {(r.CategoryName or "").lower().strip(): r.CategoryID for r in rows if r.CategoryName}
        maps["category_names"] = {r.CategoryID: r.CategoryName for r in rows if r.CategoryName}
        lists["categories"] = [r.CategoryName for r in rows if r.CategoryName]

        # Subcategories
        cursor.execute("SELECT SubCategoryID, SubCategoryName FROM dbo.APP_LOOKUP_SUBCATEGORY ORDER BY SubCategoryName")
        rows = cursor.fetchall()
        maps["subcategories"] = {(r.SubCategoryName or "").lower().strip(): r.SubCategoryID for r in rows if r.SubCategoryName}
        maps["subcategory_names"] = {r.SubCategoryID: r.SubCategoryName for r in rows if r.SubCategoryName}
        lists["subcategories"] = [r.SubCategoryName for r in rows if r.SubCategoryName]

        # Classifications + chain data (domain/category/subcategory per classification)
        cursor.execute("""
            SELECT c.ClassificationID, c.Classification_AR, c.SubCategoryID,
                   s.CategoryID, cat.DomainID
            FROM dbo.APP_LOOKUP_CLASSIFICATION c
            JOIN dbo.APP_LOOKUP_SUBCATEGORY s ON c.SubCategoryID = s.SubCategoryID
            JOIN dbo.APP_LOOKUP_CATEGORY cat ON s.CategoryID = cat.CategoryID
            WHERE c.IsActive = 1
            ORDER BY c.Classification_AR
        """)
        rows = cursor.fetchall()
        classification_map = {}
        classification_chains = {}
        classification_list = []
        for r in rows:
            name_key = (r.Classification_AR or "").lower().strip()
            if name_key:
                if name_key not in classification_map:
                    classification_map[name_key] = r.ClassificationID
                    classification_list.append(r.Classification_AR)
            classification_chains[r.ClassificationID] = {
                "domain_id": r.DomainID,
                "category_id": r.CategoryID,
                "subcategory_id": r.SubCategoryID,
            }
        maps["classifications"] = classification_map
        lists["classifications"] = classification_list
        maps["classification_chains"] = classification_chains

        # Severities
        cursor.execute("SELECT SeverityID, SeverityName FROM dbo.APP_LOOKUP_SEVERITY ORDER BY SeverityID")
        rows = cursor.fetchall()
        maps["severities"] = {(r.SeverityName or "").lower().strip(): r.SeverityID for r in rows if r.SeverityName}
        lists["severities"] = [r.SeverityName for r in rows if r.SeverityName]

        # Stages
        cursor.execute("SELECT StageID, StageName FROM dbo.APP_LOOKUP_CASE_STAGE ORDER BY StageOrder")
        rows = cursor.fetchall()
        maps["stages"] = {(r.StageName or "").lower().strip(): r.StageID for r in rows if r.StageName}
        lists["stages"] = [r.StageName for r in rows if r.StageName]

        # Harm Levels
        cursor.execute("SELECT HarmID, HarmLevel FROM dbo.APP_LOOKUP_HARM_LEVEL ORDER BY SeverityOrder")
        rows = cursor.fetchall()
        maps["harm_levels"] = {(r.HarmLevel or "").lower().strip(): r.HarmID for r in rows if r.HarmLevel}
        lists["harm_levels"] = [r.HarmLevel for r in rows if r.HarmLevel]

        # Clinical Risk Types
        cursor.execute("SELECT ClinicalRiskTypeID, Name FROM dbo.APP_LOOKUP_CLINICAL_RISK_TYPE WHERE IsActive=1 ORDER BY DisplayOrder")
        rows = cursor.fetchall()
        maps["risk_types"] = {(r.Name or "").lower().strip(): r.ClinicalRiskTypeID for r in rows if r.Name}
        lists["risk_types"] = [r.Name for r in rows if r.Name]

        # Buildings
        cursor.execute("SELECT BuildingID, BuildingName FROM dbo.APP_LOOKUP_BUILDING ORDER BY BuildingCode")
        rows = cursor.fetchall()
        maps["buildings"] = {(r.BuildingName or "").lower().strip(): r.BuildingID for r in rows if r.BuildingName}
        lists["buildings"] = [r.BuildingName for r in rows if r.BuildingName]

        # NOTE: doctors/workers are NOT loaded here. They come from the
        # merged reserve + Hospital Directory API source instead (see
        # import_service._load_directory_lookups) -- this module is pure
        # local SQL only and has no business reaching out to the external
        # directory client.

    finally:
        cursor.close()
        conn.close()

    return {"maps": maps, "lists": lists}


# ============================================================
# PATIENT MATCHING
# ============================================================
# Exact-match counting against the Hospital Directory API + reserve table
# now lives in import_service._count_patient_matches (it needs the external
# API client, which is business logic, not pure SQL -- this module stays
# SQL-only). count_patient_exact() used to query the retired
# dbo.VW_PatientAdmission view directly and crashed on any install where
# that view no longer exists.


def create_reserve_patient(full_name: str, created_by_user_id: int, cursor) -> None:
    """Insert a minimal patient record into APP_RESERVE_PATIENT."""
    # Only the first part of the name goes to FirstName; store full name in FullName
    parts = full_name.strip().split()
    first = parts[0] if parts else full_name
    rest = " ".join(parts[1:]) if len(parts) > 1 else None

    cursor.execute("""
        INSERT INTO dbo.APP_RESERVE_PATIENT
            (FirstName, MiddleName, FullName, SystemTime)
        VALUES (?, ?, ?, GETDATE())
    """, (first, rest, full_name))


# ============================================================
# WORKER MATCHING
# ============================================================
# find_worker_by_name() used to query HR_EMPLOYEES_TABLE (a stale local HR
# view) directly. Worker matching now goes through the merged reserve +
# Hospital Directory API source, same as doctors -- see
# import_service._load_directory_lookups.


# ============================================================
# INSERT OPERATIONS (all accept an open cursor for transactions)
# ============================================================

def insert_incident(
    patient_name: str,
    feedback_intent_type_id: Optional[int],
    issuing_org_unit_id: Optional[int],
    building_id: Optional[int],
    is_inpatient: bool,
    created_by_user_id: int,
    cursor,
) -> int:
    cursor.execute("""
        INSERT INTO dbo.APP_Incident
            (incident_number, patient_name, feedback_intent_type_id, issuing_org_unit_id,
             building_id, is_inpatient, created_by_user_id)
        OUTPUT INSERTED.incident_id
        VALUES (?,?,?,?,?,?,?)
    """, ("PENDING", patient_name, feedback_intent_type_id, issuing_org_unit_id,
          building_id, int(is_inpatient), created_by_user_id))
    incident_id = int(cursor.fetchone()[0])

    # incident_number is NOT NULL with no DB-side default, and incident_id
    # isn't known until after the INSERT (IDENTITY column) -- same
    # placeholder-then-overwrite pattern as incident_parent.create_incident_parent,
    # same "INC-000123" format.
    cursor.execute(
        "UPDATE dbo.APP_Incident SET incident_number = ? WHERE incident_id = ?",
        (f"INC-{incident_id:06d}", incident_id),
    )
    return incident_id


def insert_case(
    incident_id: int,
    complaint_text: str,
    immediate_action: Optional[str],
    taken_action: Optional[str],
    feedback_date: date,
    patient_name: str,
    issuing_org_unit_id: Optional[int],
    created_by_user_id: int,
    is_inpatient: bool,
    clinical_risk_type_id: Optional[int],
    feedback_intent_type_id: Optional[int],
    building_id: Optional[int],
    domain_id: Optional[int],
    category_id: Optional[int],
    subcategory_id: Optional[int],
    classification_id: Optional[int],
    severity_id: Optional[int],
    stage_id: Optional[int],
    harm_id: Optional[int],
    source_id: Optional[int],
    cursor,
) -> int:
    cursor.execute("""
        INSERT INTO dbo.APP_IncidentCase
            (incident_id, ComplaintText, ImmediateAction, TakenAction,
             FeedbackRecievedDate, PatientName, IssuingOrgUnitID,
             CreatedByUserID, isINPatient, ClinicalRiskTypeID,
             FeedbackIntentTypeID, BuildingID, DomainID, CategoryID,
             SubCategoryID, ClassificationID, SeverityID, StageID,
             HarmLevelID, CaseStatusID, SourceID, ExplanationStatusID, RequiresExplanation)
        OUTPUT INSERTED.IncidentRequestCaseID
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1,?,4,0)
    """, (
        incident_id, complaint_text, immediate_action, taken_action,
        feedback_date, patient_name, issuing_org_unit_id,
        created_by_user_id, int(is_inpatient), clinical_risk_type_id,
        feedback_intent_type_id, building_id, domain_id, category_id,
        subcategory_id, classification_id, severity_id, stage_id,
        harm_id, source_id
    ))
    return cursor.fetchone()[0]


def insert_target_dept(
    case_id: int,
    dept_id: int,
    is_primary: bool,
    assigned_by_user_id: int,
    cursor,
) -> None:
    cursor.execute("""
        INSERT INTO dbo.APP_IncidentCaseTargetDepartment
            (IncidentRequestCaseID, DepartmentID, IsPrimary, AssignedByUserID)
        VALUES (?,?,?,?)
    """, (case_id, dept_id, int(is_primary), assigned_by_user_id))


def insert_subcase_draft(
    case_id: int,
    target_org_unit_id: int,
    created_by_user_id: int,
    cursor,
) -> int:
    cursor.execute("""
        INSERT INTO dbo.APP_AdministrativeSubcase
            (CaseType, IncidentRequestCaseID, SeasonalReportID,
             TargetOrgUnitID, Status, CreatedAt, CreatedByUserID)
        OUTPUT INSERTED.SubcaseID
        VALUES ('INCIDENT_RESPONSE', ?, NULL, ?, 'DRAFT', GETDATE(), ?)
    """, (case_id, target_org_unit_id, created_by_user_id))
    return cursor.fetchone()[0]


def insert_case_doctor(
    case_id: int,
    doctor_id: int,
    doctor_name: str,
    assigned_by_user_id: int,
    cursor,
) -> None:
    cursor.execute("""
        INSERT INTO dbo.APP_IncidentCaseDoctor
            (IncidentRequestCaseID, DoctorID, DoctorName, IsPrimary, AssignedByUserID)
        VALUES (?,?,?,0,?)
    """, (case_id, doctor_id, doctor_name, assigned_by_user_id))


def insert_case_employee(
    case_id: int,
    employee_id: int,
    assigned_by_user_id: int,
    cursor,
) -> None:
    cursor.execute("""
        INSERT INTO dbo.APP_IncidentCaseEmployee
            (IncidentRequestCaseID, EmployeeID, AssignedByUserID, IsPrimary, AssignedAt)
        VALUES (?,?,?,0,GETDATE())
    """, (case_id, employee_id, assigned_by_user_id))
