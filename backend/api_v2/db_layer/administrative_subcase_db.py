"""
Administrative Subcase Database Layer (API V2)
Handles SQL operations for APP_AdministrativeSubcase table.

This is part of Phase 3 parallel workflow system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from core.database import get_connection


# ============================================================
# CREATION / FETCH
# ============================================================

def create_subcase(
    case_type: str,
    incident_id: Optional[int],
    seasonal_report_id: Optional[int],
    target_org_unit_id: int,
    created_by_user_id: int,
    initial_status: str = "SUBMITTED_TO_SECTION",
    section_deadline_at: Optional[datetime] = None,
    department_deadline_at: Optional[datetime] = None,
    administration_deadline_at: Optional[datetime] = None
) -> Optional[int]:
    """
    Create a new administrative subcase.

    Args:
        case_type: 'INCIDENT_RESPONSE' or 'SEASONAL_REPORT_RESPONSE'
        incident_id: FK to APP_IncidentCase (or None)
        seasonal_report_id: FK to APP_SeasonalOrgUnitReport (or None)
        target_org_unit_id: FK to AdminsrationUnit
        created_by_user_id: Who created this subcase
        initial_status: Initial workflow status (default: SUBMITTED_TO_SECTION)
        section_deadline_at: Initial SectionDeadlineAt (only set when the
            workflow starts at the Section level for a non-excluded case)
        department_deadline_at: Initial DepartmentDeadlineAt (only set when
            the workflow starts at the Department level for a non-excluded case)
        administration_deadline_at: Initial AdministrationDeadlineAt (only set
            when the workflow starts at the Administration level for a
            non-excluded case)

    Returns:
        SubcaseID if created, None on failure
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            INSERT INTO dbo.APP_AdministrativeSubcase (
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                CreatedAt,
                CreatedByUserID,
                SectionDeadlineAt,
                DepartmentDeadlineAt,
                AdministrationDeadlineAt
            )
            OUTPUT INSERTED.SubcaseID
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor.execute(query, (
            case_type,
            incident_id,
            seasonal_report_id,
            target_org_unit_id,
            initial_status,
            datetime.now(),
            created_by_user_id,
            section_deadline_at,
            department_deadline_at,
            administration_deadline_at
        ))

        row = cursor.fetchone()
        new_id = row[0] if row else None

        conn.commit()
        return new_id

    finally:
        cursor.close()
        conn.close()


def get_subcase_by_id(subcase_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a single subcase by ID.
    
    Returns:
        Subcase dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                SectionExplanationText,
                SectionRejectionText,
                DepartmentExplanationText,
                DepartmentRejectionText,
                AdministrationExplanationText,
                AdministrationRejectionText,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID,
                ForceClosedAt,
                ForceClosedByUserID,
                ForceCloseReason
            FROM dbo.APP_AdministrativeSubcase
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (subcase_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            "subcase_id": row.SubcaseID,
            "case_type": row.CaseType,
            "incident_request_case_id": row.IncidentRequestCaseID,
            "seasonal_report_id": row.SeasonalReportID,
            "target_org_unit_id": row.TargetOrgUnitID,
            "status": row.Status,
            "section_explanation_text": row.SectionExplanationText,
            "section_rejection_text": row.SectionRejectionText,
            "department_explanation_text": row.DepartmentExplanationText,
            "department_rejection_text": row.DepartmentRejectionText,
            "administration_explanation_text": row.AdministrationExplanationText,
            "administration_rejection_text": row.AdministrationRejectionText,
            "created_at": row.CreatedAt,
            "created_by_user_id": row.CreatedByUserID,
            "updated_at": row.UpdatedAt,
            "updated_by_user_id": row.UpdatedByUserID,
            "force_closed_at": row.ForceClosedAt,
            "force_closed_by_user_id": row.ForceClosedByUserID,
            "force_close_reason": row.ForceCloseReason
        }
    
    finally:
        cursor.close()
        conn.close()


def get_all_subcases() -> List[Dict[str, Any]]:
    """
    Fetch ALL subcases from the database (no filtering).
    Used by insight_service for scope filtering.
    
    Returns:
        List of all subcase dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                SectionExplanationText,
                SectionRejectionText,
                DepartmentExplanationText,
                DepartmentRejectionText,
                AdministrationExplanationText,
                AdministrationRejectionText,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase
            ORDER BY SubcaseID DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        subcases = []
        for row in rows:
            subcases.append({
                "SubcaseID": row.SubcaseID,
                "CaseType": row.CaseType,
                "IncidentRequestCaseID": row.IncidentRequestCaseID,
                "SeasonalReportID": row.SeasonalReportID,
                "TargetOrgUnitID": row.TargetOrgUnitID,
                "Status": row.Status,
                "SectionExplanationText": row.SectionExplanationText,
                "SectionRejectionText": row.SectionRejectionText,
                "DepartmentExplanationText": row.DepartmentExplanationText,
                "DepartmentRejectionText": row.DepartmentRejectionText,
                "AdministrationExplanationText": row.AdministrationExplanationText,
                "AdministrationRejectionText": row.AdministrationRejectionText,
                "CreatedAt": row.CreatedAt,
                "CreatedByUserID": row.CreatedByUserID,
                "UpdatedAt": row.UpdatedAt,
                "UpdatedByUserID": row.UpdatedByUserID
            })
        
        return subcases
    
    finally:
        cursor.close()
        conn.close()


def get_subcases_by_incident(incident_id: int) -> List[Dict[str, Any]]:
    """
    Fetch all subcases linked to a specific incident.
    
    Returns:
        List of subcase dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                SectionExplanationText,
                SectionRejectionText,
                DepartmentExplanationText,
                DepartmentRejectionText,
                AdministrationExplanationText,
                AdministrationRejectionText,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID = ?
            ORDER BY CreatedAt ASC
        """
        
        cursor.execute(query, (incident_id,))
        rows = cursor.fetchall()
        
        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "status": row.Status,
                "section_explanation_text": row.SectionExplanationText,
                "section_rejection_text": row.SectionRejectionText,
                "department_explanation_text": row.DepartmentExplanationText,
                "department_rejection_text": row.DepartmentRejectionText,
                "administration_explanation_text": row.AdministrationExplanationText,
                "administration_rejection_text": row.AdministrationRejectionText,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


def get_subcases_by_seasonal_report(seasonal_report_id: int) -> List[Dict[str, Any]]:
    """
    Fetch all subcases linked to a specific seasonal report.
    
    Returns:
        List of subcase dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                SectionExplanationText,
                SectionRejectionText,
                DepartmentExplanationText,
                DepartmentRejectionText,
                AdministrationExplanationText,
                AdministrationRejectionText,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase
            WHERE SeasonalReportID = ?
            ORDER BY CreatedAt ASC
        """
        
        cursor.execute(query, (seasonal_report_id,))
        rows = cursor.fetchall()
        
        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "status": row.Status,
                "section_explanation_text": row.SectionExplanationText,
                "section_rejection_text": row.SectionRejectionText,
                "department_explanation_text": row.DepartmentExplanationText,
                "department_rejection_text": row.DepartmentRejectionText,
                "administration_explanation_text": row.AdministrationExplanationText,
                "administration_rejection_text": row.AdministrationRejectionText,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


def get_subcases_by_target_orgunit(target_orgunit_id: int) -> List[Dict[str, Any]]:
    """
    Fetch all subcases targeting a specific org unit.
    
    Returns:
        List of subcase dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                SectionExplanationText,
                SectionRejectionText,
                DepartmentExplanationText,
                DepartmentRejectionText,
                AdministrationExplanationText,
                AdministrationRejectionText,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase
            WHERE TargetOrgUnitID = ?
            ORDER BY CreatedAt DESC
        """
        
        cursor.execute(query, (target_orgunit_id,))
        rows = cursor.fetchall()
        
        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "status": row.Status,
                "section_explanation_text": row.SectionExplanationText,
                "section_rejection_text": row.SectionRejectionText,
                "department_explanation_text": row.DepartmentExplanationText,
                "department_rejection_text": row.DepartmentRejectionText,
                "administration_explanation_text": row.AdministrationExplanationText,
                "administration_rejection_text": row.AdministrationRejectionText,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


def get_subcases_by_status(status_code: str) -> List[Dict[str, Any]]:
    """
    Fetch all subcases with a specific status.
    
    Returns:
        List of subcase dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.IncidentRequestCaseID,
                sub.SeasonalReportID,
                sub.TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgUnitType,
                sub.Status,
                sub.SectionExplanationText,
                sub.SectionRejectionText,
                sub.DepartmentExplanationText,
                sub.DepartmentRejectionText,
                sub.AdministrationExplanationText,
                sub.AdministrationRejectionText,
                sub.CreatedAt,
                sub.CreatedByUserID,
                sub.UpdatedAt,
                sub.UpdatedByUserID,
                inc.incident_number AS IncidentNumber,
                sub.SectionDeadlineAt,
                sub.DepartmentDeadlineAt,
                sub.AdministrationDeadlineAt,
                sub.SectionForceClosedAt,
                sub.SectionLateReply,
                sub.SectionExtraTimeGrantedAt,
                sub.DepartmentForceClosedAt,
                sub.DepartmentLateReply,
                sub.DepartmentExtraTimeGrantedAt,
                sub.AdministrationForceClosedAt,
                sub.AdministrationLateReply,
                sub.AdministrationExtraTimeGrantedAt,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,
                ic.RecordTypeID,

                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                cat.CategoryName,
                sev.SeverityName,

                -- Originating section/unit (immutable on the parent case, unlike
                -- TargetOrgUnitID which can be repointed as the subcase escalates)
                issuing_org.Name AS IssuingOrgUnitName,

                -- Patient Services decision fields (null unless decision has been recorded)
                sub.PatientServicesDecisionText

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.AdminsrationUnit issuing_org
                ON ic.IssuingOrgUnitID = issuing_org.UniqueID
            WHERE sub.Status = ?
            ORDER BY sub.CreatedAt DESC
        """

        cursor.execute(query, (status_code,))
        rows = cursor.fetchall()

        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "target_org_unit_type": row.OrgUnitType,
                "status": row.Status,
                "section_explanation_text": row.SectionExplanationText,
                "section_rejection_text": row.SectionRejectionText,
                "department_explanation_text": row.DepartmentExplanationText,
                "department_rejection_text": row.DepartmentRejectionText,
                "administration_explanation_text": row.AdministrationExplanationText,
                "administration_rejection_text": row.AdministrationRejectionText,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID,
                "incident_number": row.IncidentNumber,
                "section_deadline_at": row.SectionDeadlineAt,
                "department_deadline_at": row.DepartmentDeadlineAt,
                "administration_deadline_at": row.AdministrationDeadlineAt,
                "section_force_closed_at": row.SectionForceClosedAt,
                "section_late_reply": bool(row.SectionLateReply),
                "section_extra_time_granted_at": row.SectionExtraTimeGrantedAt,
                "department_force_closed_at": row.DepartmentForceClosedAt,
                "department_late_reply": bool(row.DepartmentLateReply),
                "department_extra_time_granted_at": row.DepartmentExtraTimeGrantedAt,
                "administration_force_closed_at": row.AdministrationForceClosedAt,
                "administration_late_reply": bool(row.AdministrationLateReply),
                "administration_extra_time_granted_at": row.AdministrationExtraTimeGrantedAt,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "record_type_id": row.RecordTypeID,
                "waiting_days": int(row.WaitingDays or 0),
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "category": row.CategoryName,
                "severity": row.SeverityName or "NEUTRAL",
                "issuing_org_unit_name": row.IssuingOrgUnitName,
                "patient_services_decision_text": row.PatientServicesDecisionText,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()


def get_subcases_by_case_type(case_type: str) -> List[Dict[str, Any]]:
    """
    Fetch all subcases of a specific type.
    
    Args:
        case_type: 'INCIDENT_RESPONSE' or 'SEASONAL_REPORT_RESPONSE'
    
    Returns:
        List of subcase dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                SeasonalReportID,
                TargetOrgUnitID,
                Status,
                SectionExplanationText,
                SectionRejectionText,
                DepartmentExplanationText,
                DepartmentRejectionText,
                AdministrationExplanationText,
                AdministrationRejectionText,
                CreatedAt,
                CreatedByUserID,
                UpdatedAt,
                UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase
            WHERE CaseType = ?
            ORDER BY CreatedAt DESC
        """
        
        cursor.execute(query, (case_type,))
        rows = cursor.fetchall()
        
        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "status": row.Status,
                "section_explanation_text": row.SectionExplanationText,
                "section_rejection_text": row.SectionRejectionText,
                "department_explanation_text": row.DepartmentExplanationText,
                "department_rejection_text": row.DepartmentRejectionText,
                "administration_explanation_text": row.AdministrationExplanationText,
                "administration_rejection_text": row.AdministrationRejectionText,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID
            }
            for row in rows
        ]
    
    finally:
        cursor.close()
        conn.close()


# ============================================================
# INBOX QUERIES (Role-based filtering done in service layer)
# ============================================================

def get_subcases_denied_by_section() -> List[Dict[str, Any]]:
    """
    Fetch subcases denied by section.
    Status = 'SECTION_DENIED'
    
    These are cases where the section administrator rejected responsibility.
    The COMPLAINT_SUPERVISOR can see these and reopen them.
    
    Returns:
        List of subcase dicts
    """
    return get_subcases_by_status("SECTION_DENIED")


def get_subcases_pending_for_section() -> List[Dict[str, Any]]:
    """
    Fetch subcases pending section response.
    Status = 'SUBMITTED_TO_SECTION' OR 'RETURNED_TO_SECTION_FOR_REVISION'
    
    Returns:
        List of subcase dicts
    """
    # Get both initial submissions and returned-for-revision cases
    initial = get_subcases_by_status("SUBMITTED_TO_SECTION")
    returned = get_subcases_by_status("RETURNED_TO_SECTION_FOR_REVISION")
    return initial + returned


def get_subcases_pending_for_department() -> List[Dict[str, Any]]:
    """
    Fetch subcases pending department response.
    Status = 'SECTION_ACCEPTED_PENDING_DEPT' OR 'RETURNED_TO_DEPT_FOR_REVISION'
    OR 'FORCE_CLOSED_AT_SECTION' (Section missed its deadline and the case
    escalated to Department responsibility)

    Returns:
        List of subcase dicts
    """
    # Get initial submissions, returned-for-revision cases, and cases
    # escalated to Department because Section was automatically force-closed
    initial = get_subcases_by_status("SECTION_ACCEPTED_PENDING_DEPT")
    returned = get_subcases_by_status("RETURNED_TO_DEPT_FOR_REVISION")
    escalated = get_subcases_by_status("FORCE_CLOSED_AT_SECTION")
    return initial + returned + escalated


def get_subcases_pending_for_administration() -> List[Dict[str, Any]]:
    """
    Fetch subcases pending administration response.
    Status = 'DEPT_ACCEPTED_PENDING_ADMIN' OR 'FORCE_CLOSED_AT_DEPARTMENT'
    (Department missed its deadline and the case escalated to
    Administration responsibility)

    Returns:
        List of subcase dicts
    """
    initial = get_subcases_by_status("DEPT_ACCEPTED_PENDING_ADMIN")
    escalated = get_subcases_by_status("FORCE_CLOSED_AT_DEPARTMENT")
    return initial + escalated


def get_subcases_waiting_patient_services_decision() -> List[Dict[str, Any]]:
    """
    Fetch subcases waiting for the Patient Services scientific decision.
    Status = 'WAITING_PATIENT_SERVICES_DECISION'

    These are administrative complaint subcases where Administration has
    approved but the Complaint Supervisor still needs to record the
    قرار خدمات المرضى بحسب المراجع العلميّة.

    Returns:
        List of subcase dicts
    """
    return get_subcases_by_status("WAITING_PATIENT_SERVICES_DECISION")


def save_patient_services_decision(
    subcase_id: int,
    decision_text: str,
    user_id: int
) -> bool:
    """
    Persist the Patient Services scientific decision on a subcase.

    - Always writes decision_text, user_id, updated_at.
    - Sets decision_at only on the first save (preserves original timestamp on edits).
    - Transitions subcase status to PATIENT_SERVICES_DECISION_COMPLETED.
    - Does NOT touch action items.

    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        now = datetime.now()
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET PatientServicesDecisionText      = ?,
                PatientServicesDecisionByUserID  = ?,
                PatientServicesDecisionAt        = CASE
                    WHEN PatientServicesDecisionAt IS NULL THEN ?
                    ELSE PatientServicesDecisionAt
                END,
                PatientServicesDecisionUpdatedAt = ?,
                Status           = 'PATIENT_SERVICES_DECISION_COMPLETED',
                UpdatedAt        = ?,
                UpdatedByUserID  = ?
            WHERE SubcaseID = ?
        """
        cursor.execute(query, (
            decision_text,
            user_id,
            now,   # PatientServicesDecisionAt (first save only)
            now,   # PatientServicesDecisionUpdatedAt (always)
            now,   # UpdatedAt
            user_id,
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0

    finally:
        cursor.close()
        conn.close()


def acknowledge_decision_notification(
    subcase_id: int,
    user_id: int
) -> bool:
    """
    Acknowledge a completed Patient Services decision.

    Transitions status from PATIENT_SERVICES_DECISION_COMPLETED to
    DECISION_ACKNOWLEDGED. The status guard prevents double-acknowledging
    or acting on wrong-status subcases.

    Returns:
        True if updated, False if subcase not found or already acknowledged
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        cursor.execute(
            """
            UPDATE dbo.APP_AdministrativeSubcase
            SET Status          = 'DECISION_ACKNOWLEDGED',
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND Status    = 'PATIENT_SERVICES_DECISION_COMPLETED'
            """,
            (now, user_id, subcase_id)
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def get_decision_acknowledgment_levels(subcase_id: int) -> set:
    """Return the set of OrgLevel values already acknowledged for this subcase."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT OrgLevel FROM dbo.APP_SubcaseDecisionAcknowledgment WHERE SubcaseID = ?",
            subcase_id
        )
        return {row.OrgLevel for row in cursor.fetchall()}
    finally:
        cursor.close()
        conn.close()


def get_acknowledged_subcase_ids_for_level(level: str) -> set:
    """Return the set of SubcaseIDs already acknowledged by the given OrgLevel — bulk lookup for inbox filtering."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT SubcaseID FROM dbo.APP_SubcaseDecisionAcknowledgment WHERE OrgLevel = ?",
            level
        )
        return {row.SubcaseID for row in cursor.fetchall()}
    finally:
        cursor.close()
        conn.close()


def insert_decision_acknowledgment(subcase_id: int, level: str, user_id: int) -> None:
    """Record that `level` has acknowledged the completed decision on this subcase."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            INSERT INTO dbo.APP_SubcaseDecisionAcknowledgment
                (SubcaseID, OrgLevel, AcknowledgedByUserID, AcknowledgedAt)
            VALUES (?, ?, ?, ?)
            """,
            (subcase_id, level, user_id, datetime.now())
        )
        conn.commit()
    finally:
        cursor.close()
        conn.close()


# ============================================================
# WORKFLOW MUTATION HELPERS (NO validation logic)
# ============================================================

def update_subcase_status(
    subcase_id: int,
    new_status: str,
    updated_by_user_id: int
) -> bool:
    """
    Update subcase status.
    NO validation. Service layer handles workflow rules.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET Status = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            new_status,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_section_explanation(
    subcase_id: int,
    text: str,
    updated_by_user_id: int
) -> bool:
    """
    Update section explanation text.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET SectionExplanationText = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            text,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_section_rejection(
    subcase_id: int,
    text: str,
    updated_by_user_id: int
) -> bool:
    """
    Update section rejection text.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET SectionRejectionText = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            text,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_department_explanation(
    subcase_id: int,
    text: str,
    updated_by_user_id: int
) -> bool:
    """
    Update department explanation text.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET DepartmentExplanationText = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            text,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_department_rejection(
    subcase_id: int,
    text: str,
    updated_by_user_id: int
) -> bool:
    """
    Update department rejection text.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET DepartmentRejectionText = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            text,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_administration_explanation(
    subcase_id: int,
    text: str,
    updated_by_user_id: int
) -> bool:
    """
    Update administration explanation text.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET AdministrationExplanationText = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            text,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def update_administration_rejection(
    subcase_id: int,
    text: str,
    updated_by_user_id: int
) -> bool:
    """
    Update administration rejection text.
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET AdministrationRejectionText = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            text,
            datetime.now(),
            updated_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


# ============================================================
# MONITORING / INSIGHT
# ============================================================

def get_full_subcases_by_incident(incident_id: int) -> List[Dict[str, Any]]:
    """
    Fetch complete subcases for an incident (includes all fields).
    Same as get_subcases_by_incident but with explicit "full" naming.
    
    Returns:
        List of subcase dicts with all fields
    """
    return get_subcases_by_incident(incident_id)


def get_full_subcases_by_seasonal_report(seasonal_report_id: int) -> List[Dict[str, Any]]:
    """
    Fetch complete subcases for a seasonal report (includes all fields).
    Same as get_subcases_by_seasonal_report but with explicit "full" naming.
    
    Returns:
        List of subcase dicts with all fields
    """
    return get_subcases_by_seasonal_report(seasonal_report_id)


# ============================================================
# FORCE CLOSE TRACKING
# ============================================================

def update_force_close_tracking(
    subcase_id: int,
    force_closed_by_user_id: int,
    force_close_reason: str
) -> bool:
    """
    Update force close tracking fields for a subcase.
    
    Sets ForceClosedAt, ForceClosedByUserID, and ForceCloseReason.
    Does NOT update Status — use force_close_subcase_with_tracking for that.
    
    Args:
        subcase_id: Subcase ID
        force_closed_by_user_id: User ID who force closed the subcase
        force_close_reason: Reason for force closing
    
    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET ForceClosedAt = ?,
                ForceClosedByUserID = ?,
                ForceCloseReason = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            datetime.now(),
            force_closed_by_user_id,
            force_close_reason,
            datetime.now(),
            force_closed_by_user_id,
            subcase_id
        ))
        
        conn.commit()
        return cursor.rowcount > 0
    
    finally:
        cursor.close()
        conn.close()


def force_close_subcase_with_tracking(
    subcase_id: int,
    force_closed_by_user_id: int,
    force_close_reason: str,
    new_status: str
) -> bool:
    """
    Force close a subcase with full tracking (status + tracking fields).

    Caller must pass new_status explicitly — either 'FORCE_CLOSED_DRAFT'
    (data incomplete) or 'FORCE_CLOSED_COMPLETE' (all data present).

    Args:
        subcase_id: Subcase ID
        force_closed_by_user_id: User ID who force closed the subcase
        force_close_reason: Reason for force closing
        new_status: Target status ('FORCE_CLOSED_DRAFT' or 'FORCE_CLOSED_COMPLETE')

    Returns:
        True if updated, False if subcase not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET Status = ?,
                ForceClosedAt = ?,
                ForceClosedByUserID = ?,
                ForceCloseReason = ?,
                UpdatedAt = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
        """
        
        cursor.execute(query, (
            new_status,
            datetime.now(),
            force_closed_by_user_id,
            force_close_reason,
            datetime.now(),
            force_closed_by_user_id,
            subcase_id
        ))

        conn.commit()
        return cursor.rowcount > 0

    finally:
        cursor.close()
        conn.close()


# ============================================================
# ENHANCED INBOX QUERIES WITH FULL DETAILS (for Insight page)
# ============================================================

def get_subcases_with_details_for_section() -> List[Dict[str, Any]]:
    """
    Fetch subcases for section admin WITH full details for Insight page.
    
    Includes:
    - Case description (FeedbackText)
    - Patient name
    - Severity
    - Category
    - Org unit name
    - Waiting days
    
    Returns only SUBMITTED_TO_SECTION and RETURNED_TO_SECTION_FOR_REVISION
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,
                
                -- Target Org Unit (the original section always)
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,

                -- Responsible Org Unit: for section-stage cases the section IS the responsible entity
                org.UniqueID AS ResponsibleOrgUnitID,
                org.Name AS ResponsibleOrgUnitName,
                org.Type AS ResponsibleOrgType,

                -- Incident Case Info (for INCIDENT_RESPONSE)
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,

                -- Incident parent number (INC-XXXXXX)
                inc.incident_number AS IncidentNumber,

                -- Clinical Risk Type (for Red Flag / Never Event badges)
                ic.ClinicalRiskTypeID,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,

                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName,

                -- Time-distinction fields (Session 9B) — section is the acting level here
                sub.SectionDeadlineAt,
                sub.SectionExtraTimeGrantedAt,
                sub.SectionLateReply

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID

            WHERE sub.Status IN ('SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION')
              AND sub.Status NOT IN ('FORCE_CLOSED', 'FORCE_CLOSED_DRAFT', 'FORCE_CLOSED_COMPLETE')

            ORDER BY WaitingDays DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "status": row.Status,
                "created_at": row.CreatedAt,
                "waiting_days": row.WaitingDays,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "org_type": row.OrgType,
                "responsible_org_unit_id": row.ResponsibleOrgUnitID,
                "responsible_org_unit_name": row.ResponsibleOrgUnitName,
                "responsible_org_type": row.ResponsibleOrgType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "incident_number": row.IncidentNumber,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName,
                "deadline_at": row.SectionDeadlineAt,
                "extra_time_granted_at": row.SectionExtraTimeGrantedAt,
                "is_late": bool(row.SectionLateReply),
            })

        return result

    finally:
        cursor.close()
        conn.close()


def get_subcases_with_details_for_department() -> List[Dict[str, Any]]:
    """
    Fetch subcases for department admin WITH full details for Insight page.

    Filters for SECTION_ACCEPTED_PENDING_DEPT and RETURNED_TO_DEPT_FOR_REVISION.

    The responsible org unit is the DEPARTMENT (parent of the target section),
    because the section has already responded and the department now owns the case.
    The target section is preserved for scope filtering; the parent department is
    used for Insight grouping and supervisor lookup.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,

                -- Target Org Unit (the original section — used for scope filtering)
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,

                -- Responsible Org Unit: the department (parent of the section)
                -- This is the entity that currently owns the pending response.
                dept.UniqueID AS ResponsibleOrgUnitID,
                dept.Name AS ResponsibleOrgUnitName,
                dept.Type AS ResponsibleOrgType,

                -- Incident Case Info (for INCIDENT_RESPONSE)
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,

                -- Incident parent number (INC-XXXXXX)
                inc.incident_number AS IncidentNumber,

                -- Clinical Risk Type (for Red Flag / Never Event badges)
                ic.ClinicalRiskTypeID,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,

                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName,

                -- Time-distinction fields (Session 9B) — department is the acting level here
                sub.DepartmentDeadlineAt,
                sub.DepartmentExtraTimeGrantedAt,
                sub.DepartmentLateReply

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            -- Join parent to resolve the responsible department
            LEFT JOIN dbo.AdminsrationUnit dept
                ON org.ParentID = dept.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID

            WHERE sub.Status IN ('SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION')
              AND sub.Status NOT IN ('FORCE_CLOSED', 'FORCE_CLOSED_DRAFT', 'FORCE_CLOSED_COMPLETE')

            ORDER BY WaitingDays DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "status": row.Status,
                "created_at": row.CreatedAt,
                "waiting_days": row.WaitingDays,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "org_type": row.OrgType,
                "responsible_org_unit_id": row.ResponsibleOrgUnitID,
                "responsible_org_unit_name": row.ResponsibleOrgUnitName,
                "responsible_org_type": row.ResponsibleOrgType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "incident_number": row.IncidentNumber,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName,
                "deadline_at": row.DepartmentDeadlineAt,
                "extra_time_granted_at": row.DepartmentExtraTimeGrantedAt,
                "is_late": bool(row.DepartmentLateReply),
            })

        return result

    finally:
        cursor.close()
        conn.close()


def get_subcases_with_details_for_administration() -> List[Dict[str, Any]]:
    """
    Fetch subcases for administration admin WITH full details for Insight page.

    Filters for DEPT_ACCEPTED_PENDING_ADMIN.

    The responsible org unit is the ADMINISTRATION (grandparent of the target section:
    section → department → administration), because both section and department have
    already responded and the administration now owns the pending response.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,

                -- Target Org Unit (the original section — used for scope filtering)
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,

                -- Responsible Org Unit: the administration (grandparent of the section)
                -- section.ParentID → department → department.ParentID → administration
                admin_unit.UniqueID AS ResponsibleOrgUnitID,
                admin_unit.Name AS ResponsibleOrgUnitName,
                admin_unit.Type AS ResponsibleOrgType,

                -- Incident Case Info (for INCIDENT_RESPONSE)
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,

                -- Incident parent number (INC-XXXXXX)
                inc.incident_number AS IncidentNumber,

                -- Clinical Risk Type (for Red Flag / Never Event badges)
                ic.ClinicalRiskTypeID,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,

                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName,

                -- Originating section/unit (immutable on the parent case)
                issuing_org.Name AS IssuingOrgUnitName,

                -- Time-distinction fields (Session 9B) — administration is the acting level here
                sub.AdministrationDeadlineAt,
                sub.AdministrationExtraTimeGrantedAt,
                sub.AdministrationLateReply

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            -- First hop: section → department
            LEFT JOIN dbo.AdminsrationUnit dept
                ON org.ParentID = dept.UniqueID
            -- Second hop: department → administration
            LEFT JOIN dbo.AdminsrationUnit admin_unit
                ON dept.ParentID = admin_unit.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID
            LEFT JOIN dbo.AdminsrationUnit issuing_org
                ON ic.IssuingOrgUnitID = issuing_org.UniqueID

            WHERE sub.Status = 'DEPT_ACCEPTED_PENDING_ADMIN'
              AND sub.Status NOT IN ('FORCE_CLOSED', 'FORCE_CLOSED_DRAFT', 'FORCE_CLOSED_COMPLETE')

            ORDER BY WaitingDays DESC
        """

        cursor.execute(query)
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "status": row.Status,
                "created_at": row.CreatedAt,
                "waiting_days": row.WaitingDays,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "org_type": row.OrgType,
                "responsible_org_unit_id": row.ResponsibleOrgUnitID,
                "responsible_org_unit_name": row.ResponsibleOrgUnitName,
                "responsible_org_type": row.ResponsibleOrgType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "incident_number": row.IncidentNumber,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName,
                "issuing_org_unit_name": row.IssuingOrgUnitName,
                "deadline_at": row.AdministrationDeadlineAt,
                "extra_time_granted_at": row.AdministrationExtraTimeGrantedAt,
                "is_late": bool(row.AdministrationLateReply),
            })

        return result

    finally:
        cursor.close()
        conn.close()


def get_subcases_by_statuses(status_codes: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch all subcases with any of the given statuses.
    
    Used for archive queries where we need cases that have moved past
    a particular workflow stage.
    
    Args:
        status_codes: List of status codes to include
        
    Returns:
        List of subcase dicts ordered by UpdatedAt DESC (most recently processed first)
    """
    if not status_codes:
        return []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build parameterized IN clause
        placeholders = ','.join(['?' for _ in status_codes])
        query = f"""
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.IncidentRequestCaseID,
                sub.SeasonalReportID,
                sub.TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgUnitType,
                sub.Status,
                sub.SectionExplanationText,
                sub.SectionRejectionText,
                sub.DepartmentExplanationText,
                sub.DepartmentRejectionText,
                sub.AdministrationExplanationText,
                sub.AdministrationRejectionText,
                sub.CreatedAt,
                sub.CreatedByUserID,
                sub.UpdatedAt,
                sub.UpdatedByUserID,
                inc.incident_number AS IncidentNumber,
                sub.SectionDeadlineAt,
                sub.DepartmentDeadlineAt,
                sub.AdministrationDeadlineAt,
                sub.SectionForceClosedAt,
                sub.SectionLateReply,
                sub.SectionExtraTimeGrantedAt,
                sub.DepartmentForceClosedAt,
                sub.DepartmentLateReply,
                sub.DepartmentExtraTimeGrantedAt,
                sub.AdministrationForceClosedAt,
                sub.AdministrationLateReply,
                sub.AdministrationExtraTimeGrantedAt,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,
                ic.RecordTypeID,
                sub.PatientServicesDecisionText,

                -- The real "who should solve this" target is TargetOrgUnitID
                -- (org.Name above). IssuingOrgUnitID is the separate, distinct
                -- "where this happened / who reported it" source unit.
                issuing_org.Name AS IssuingOrgUnitName

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.AdminsrationUnit issuing_org
                ON ic.IssuingOrgUnitID = issuing_org.UniqueID
            WHERE sub.Status IN ({placeholders})
            ORDER BY sub.UpdatedAt DESC
        """

        cursor.execute(query, status_codes)
        rows = cursor.fetchall()

        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "seasonal_report_id": row.SeasonalReportID,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "target_org_unit_type": row.OrgUnitType,
                "status": row.Status,
                "section_explanation_text": row.SectionExplanationText,
                "section_rejection_text": row.SectionRejectionText,
                "department_explanation_text": row.DepartmentExplanationText,
                "department_rejection_text": row.DepartmentRejectionText,
                "administration_explanation_text": row.AdministrationExplanationText,
                "administration_rejection_text": row.AdministrationRejectionText,
                "created_at": row.CreatedAt,
                "created_by_user_id": row.CreatedByUserID,
                "updated_at": row.UpdatedAt,
                "updated_by_user_id": row.UpdatedByUserID,
                "incident_number": row.IncidentNumber,
                "section_deadline_at": row.SectionDeadlineAt,
                "department_deadline_at": row.DepartmentDeadlineAt,
                "administration_deadline_at": row.AdministrationDeadlineAt,
                "section_force_closed_at": row.SectionForceClosedAt,
                "section_late_reply": bool(row.SectionLateReply),
                "section_extra_time_granted_at": row.SectionExtraTimeGrantedAt,
                "department_force_closed_at": row.DepartmentForceClosedAt,
                "department_late_reply": bool(row.DepartmentLateReply),
                "department_extra_time_granted_at": row.DepartmentExtraTimeGrantedAt,
                "administration_force_closed_at": row.AdministrationForceClosedAt,
                "administration_late_reply": bool(row.AdministrationLateReply),
                "administration_extra_time_granted_at": row.AdministrationExtraTimeGrantedAt,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "record_type_id": row.RecordTypeID,
                "patient_services_decision_text": row.PatientServicesDecisionText,
                "issuing_org_unit_name": row.IssuingOrgUnitName,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()


def get_force_closed_pipeline_cases() -> List[Dict[str, Any]]:
    """
    Fetch FORCE_CLOSED_AT_ADMINISTRATION subcases with full display details for
    the Insight page "Complaints" panel.

    AT_SECTION and AT_DEPARTMENT cases are no longer returned here — they are
    merged directly into the Department/Administration groups of
    get_grouped_inbox_for_admin() (see get_force_closed_section_cases_for_department
    and get_force_closed_department_cases_for_administration below), since
    administration is the top of the org hierarchy and AT_ADMINISTRATION cases
    have nowhere further to escalate — they stay a Complaint Supervisor concern.

    Returns patient name, description, severity, waiting days — the same fields
    used by the grouped inbox view — so the frontend can render rich cards.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,

                org.UniqueID  AS TargetOrgUnitID,
                org.Name      AS OrgUnitName,
                org.Type      AS OrgType,

                ic.IncidentRequestCaseID,
                ic.ComplaintText  AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,

                inc.incident_number AS IncidentNumber,

                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,

                sr.SeasonalReportID,

                -- Originating section/unit (immutable on the parent case — TargetOrgUnitID
                -- has already been repointed to the Administration by this stage)
                issuing_org.Name AS IssuingOrgUnitName,

                sub.AdministrationDeadlineAt,
                sub.AdministrationExtraTimeGrantedAt,
                sub.AdministrationExtraTimeGrantedBy

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.AdminsrationUnit issuing_org
                ON ic.IssuingOrgUnitID = issuing_org.UniqueID

            WHERE sub.Status = 'FORCE_CLOSED_AT_ADMINISTRATION'
            ORDER BY sub.CreatedAt ASC
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        return [
            {
                "subcase_id":              row.SubcaseID,
                "case_type":               row.CaseType,
                "status":                  row.Status,
                "created_at":              row.CreatedAt,
                "waiting_days":            int(row.WaitingDays or 0),
                "target_org_unit_id":      row.TargetOrgUnitID,
                "org_unit_name":           row.OrgUnitName or "",
                "org_type":                row.OrgType or "",
                "incident_request_case_id": row.IncidentRequestCaseID,
                "case_description":        row.CaseDescription or "",
                "patient_name":            row.PatientName or "",
                "severity_id":             row.SeverityID,
                "severity":                row.SeverityName or "NEUTRAL",
                "category":                row.CategoryName or "",
                "incident_number":         row.IncidentNumber,
                "is_red_flag":             bool(row.IsRedFlag),
                "is_never_event":          bool(row.IsNeverEvent),
                "seasonal_report_id":      row.SeasonalReportID,
                "issuing_org_unit_name":   row.IssuingOrgUnitName,
                "administration_deadline_at":            row.AdministrationDeadlineAt,
                "administration_extra_time_granted_at":  row.AdministrationExtraTimeGrantedAt,
                "administration_extra_time_granted_by":  row.AdministrationExtraTimeGrantedBy,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()


def get_force_closed_section_cases_for_department() -> List[Dict[str, Any]]:
    """
    Fetch FORCE_CLOSED_AT_SECTION subcases with the same field shape as
    get_subcases_with_details_for_department(), so they merge directly into
    the Department group of the Insight grouped inbox.

    The department now owns the pending response (accept/override/reject or
    give the section more time), so responsible_org_unit = the section's
    parent department — same join pattern as the department-stage query.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,

                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,

                dept.UniqueID AS ResponsibleOrgUnitID,
                dept.Name AS ResponsibleOrgUnitName,
                dept.Type AS ResponsibleOrgType,

                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,

                inc.incident_number AS IncidentNumber,

                ic.ClinicalRiskTypeID,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,

                sr.SeasonalReportID,
                s.SeasonName,

                sub.SectionDeadlineAt,
                sub.SectionExtraTimeGrantedAt,
                sub.SectionLateReply

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.AdminsrationUnit dept
                ON org.ParentID = dept.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID

            WHERE sub.Status = 'FORCE_CLOSED_AT_SECTION'

            ORDER BY WaitingDays DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "status": row.Status,
                "created_at": row.CreatedAt,
                "waiting_days": row.WaitingDays,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "org_type": row.OrgType,
                "responsible_org_unit_id": row.ResponsibleOrgUnitID,
                "responsible_org_unit_name": row.ResponsibleOrgUnitName,
                "responsible_org_type": row.ResponsibleOrgType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "incident_number": row.IncidentNumber,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName,
                "deadline_at": row.SectionDeadlineAt,
                "extra_time_granted_at": row.SectionExtraTimeGrantedAt,
                "is_late": bool(row.SectionLateReply),
                "is_force_closed": True,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()


def get_force_closed_department_cases_for_administration() -> List[Dict[str, Any]]:
    """
    Fetch FORCE_CLOSED_AT_DEPARTMENT subcases with the same field shape as
    get_subcases_with_details_for_administration(), so they merge directly
    into the Administration group of the Insight grouped inbox.

    The administration now owns the pending response (accept/override/reject
    or give the department more time), so responsible_org_unit = the
    grandparent administration — same two-hop join pattern as the
    administration-stage query.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,

                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,

                admin_unit.UniqueID AS ResponsibleOrgUnitID,
                admin_unit.Name AS ResponsibleOrgUnitName,
                admin_unit.Type AS ResponsibleOrgType,

                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,

                inc.incident_number AS IncidentNumber,

                ic.ClinicalRiskTypeID,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
                ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
                ic.FeedbackRecievedDate,
                ic.IncidentDate,

                sr.SeasonalReportID,
                s.SeasonName,

                sub.DepartmentDeadlineAt,
                sub.DepartmentExtraTimeGrantedAt,
                sub.DepartmentLateReply

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.AdminsrationUnit dept
                ON org.ParentID = dept.UniqueID
            LEFT JOIN dbo.AdminsrationUnit admin_unit
                ON dept.ParentID = admin_unit.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID

            WHERE sub.Status = 'FORCE_CLOSED_AT_DEPARTMENT'

            ORDER BY WaitingDays DESC
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        return [
            {
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "status": row.Status,
                "created_at": row.CreatedAt,
                "waiting_days": row.WaitingDays,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "org_type": row.OrgType,
                "responsible_org_unit_id": row.ResponsibleOrgUnitID,
                "responsible_org_unit_name": row.ResponsibleOrgUnitName,
                "responsible_org_type": row.ResponsibleOrgType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "incident_number": row.IncidentNumber,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
                "is_red_flag": bool(row.IsRedFlag),
                "is_never_event": bool(row.IsNeverEvent),
                "is_morbidity": bool(row.IsMorbidity),
                "feedback_received_date": row.FeedbackRecievedDate,
                "incident_date": row.IncidentDate,
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName,
                "deadline_at": row.DepartmentDeadlineAt,
                "extra_time_granted_at": row.DepartmentExtraTimeGrantedAt,
                "is_late": bool(row.DepartmentLateReply),
                "is_force_closed": True,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()


# ============================================================
# ARCHIVE QUERIES (Cases that passed through a workflow stage)
# ============================================================

def get_subcases_archived_for_section() -> List[Dict[str, Any]]:
    """
    Fetch subcases that have moved past the section stage.

    These are cases the section admin processed (accepted or rejected).
    Statuses: All statuses that come AFTER section stage:
    - SECTION_ACCEPTED_PENDING_DEPT: Section approved, now at department
    - RETURNED_TO_DEPT_FOR_REVISION: Department sent back (section still processed it)
    - DEPT_ACCEPTED_PENDING_ADMIN: Department approved, now at admin
    - ADMIN_APPROVED: Final approval (workflow complete)
    - SECTION_DENIED: Section rejected (terminal)
    - FORCE_CLOSED: Legacy forcibly closed (terminal)
    - FORCE_CLOSED_DRAFT: Force closed, data not yet complete
    - FORCE_CLOSED_COMPLETE: Force closed, all data filled (terminal)
    - DECISION_ACKNOWLEDGED: Section acknowledged a Patient Services decision (terminal)

    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "SECTION_ACCEPTED_PENDING_DEPT",
        "RETURNED_TO_DEPT_FOR_REVISION",
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED",
        "SECTION_DENIED",
        "FORCE_CLOSED",
        "FORCE_CLOSED_DRAFT",
        "FORCE_CLOSED_COMPLETE",
        "FORCE_CLOSED_AT_SECTION",
        "DECISION_ACKNOWLEDGED",
    ]
    return get_subcases_by_statuses(archive_statuses)


def get_subcases_archived_for_department() -> List[Dict[str, Any]]:
    """
    Fetch subcases that have moved past the department stage.

    These are cases the department admin processed (accepted or rejected).
    Statuses: All statuses that come AFTER department stage:
    - DEPT_ACCEPTED_PENDING_ADMIN: Department approved, now at admin
    - ADMIN_APPROVED: Final approval (workflow complete)
    - RETURNED_TO_SECTION_FOR_REVISION: Dept sent back to section (still dept processed it)
    - FORCE_CLOSED: Legacy forcibly closed (terminal)
    - FORCE_CLOSED_DRAFT: Force closed, data not yet complete
    - FORCE_CLOSED_COMPLETE: Force closed, all data filled (terminal)
    - DECISION_ACKNOWLEDGED: Dept-level user acknowledged a Patient Services decision (terminal)

    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED",
        "RETURNED_TO_SECTION_FOR_REVISION",
        "FORCE_CLOSED",
        "FORCE_CLOSED_DRAFT",
        "FORCE_CLOSED_COMPLETE",
        "DECISION_ACKNOWLEDGED",
    ]
    return get_subcases_by_statuses(archive_statuses)


def get_subcases_archived_for_administration() -> List[Dict[str, Any]]:
    """
    Fetch subcases that have moved past the administration stage.

    These are cases the admin processed (approved or force-closed).
    Statuses: Terminal statuses:
    - ADMIN_APPROVED: Final approval (workflow complete)
    - FORCE_CLOSED: Legacy forcibly closed (terminal)
    - FORCE_CLOSED_DRAFT: Force closed, data not yet complete
    - FORCE_CLOSED_COMPLETE: Force closed, all data filled (terminal)
    - RETURNED_TO_DEPT_FOR_REVISION: Admin sent back to department (still admin processed it)
    - DECISION_ACKNOWLEDGED: Admin-level user acknowledged a Patient Services decision (terminal)

    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "ADMIN_APPROVED",
        "FORCE_CLOSED",
        "FORCE_CLOSED_DRAFT",
        "FORCE_CLOSED_COMPLETE",
        "RETURNED_TO_DEPT_FOR_REVISION",
        "DECISION_ACKNOWLEDGED",
    ]
    return get_subcases_by_statuses(archive_statuses)


def get_subcases_archived_for_complaint_supervisor() -> List[Dict[str, Any]]:
    """
    Fetch subcases that were reopened after section denial.

    These are cases the complaint supervisor/worker reopened.
    Statuses: Cases that moved from SECTION_DENIED back to section:
    - RETURNED_TO_SECTION_FOR_REVISION: Case was reopened
    - SECTION_ACCEPTED_PENDING_DEPT: Section then approved (workflow continued)
    - DEPT_ACCEPTED_PENDING_ADMIN: Department then approved
    - ADMIN_APPROVED: Final approval
    - PATIENT_SERVICES_DECISION_COMPLETED: Supervisor recorded decision, target office not yet acknowledged
    - DECISION_ACKNOWLEDGED: Target office acknowledged the completed decision

    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "RETURNED_TO_SECTION_FOR_REVISION",
        "SECTION_ACCEPTED_PENDING_DEPT",
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED",
        "PATIENT_SERVICES_DECISION_COMPLETED",
        "DECISION_ACKNOWLEDGED",
    ]
    return get_subcases_by_statuses(archive_statuses)


def check_user_has_subcase_for_incident(incident_id: int, allowed_unit_ids: set) -> bool:
    """
    Check if a user has any subcase assigned to their org units for a given incident.
    
    Used for authorization when viewing incident details from workflow inbox.
    
    Args:
        incident_id: Incident ID to check
        allowed_unit_ids: Set of org unit IDs the user has access to
    
    Returns:
        True if user has at least one subcase for this incident in their scope, False otherwise
    """
    if not allowed_unit_ids:
        return False
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build IN clause for allowed unit IDs
        placeholders = ",".join(["?" for _ in allowed_unit_ids])
        
        query = f"""
            SELECT COUNT(*) as cnt
            FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID = ?
              AND TargetOrgUnitID IN ({placeholders})
        """
        
        params = [incident_id] + list(allowed_unit_ids)
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        return row[0] > 0 if row else False
    
    finally:
        cursor.close()
        conn.close()


# ============================================================
# MANUAL INTERVENTION: ON-BEHALF FILLS (ownership tracked)
# ============================================================

def fill_section_on_behalf(
    subcase_id: int,
    text: str,
    entered_by_user_id: int,
    entered_for_role: str,
    entry_mode: str
) -> bool:
    """
    Write section explanation text with full ownership tracking.
    Used by COMPLAINT_SUPERVISOR / WORKER filling on behalf of SECTION_ADMIN.

    entry_mode: 'ON_BEHALF' (active subcase) or 'FORCE_CLOSE_INTERVENTION' (force-closed)

    Returns True if updated, False if subcase not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET SectionExplanationText    = ?,
                SectionEnteredByUserID    = ?,
                SectionEnteredForRole     = ?,
                SectionEntryMode          = ?,
                SectionEntryTimestamp     = ?,
                UpdatedAt                 = ?,
                UpdatedByUserID           = ?
            WHERE SubcaseID = ?
        """
        now = datetime.now()
        cursor.execute(query, (
            text,
            entered_by_user_id,
            entered_for_role,
            entry_mode,
            now,
            now,
            entered_by_user_id,
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def fill_department_on_behalf(
    subcase_id: int,
    text: str,
    entered_by_user_id: int,
    entered_for_role: str,
    entry_mode: str
) -> bool:
    """
    Write department explanation text with full ownership tracking.
    Used by COMPLAINT_SUPERVISOR / WORKER filling on behalf of DEPARTMENT_ADMIN.

    entry_mode: 'ON_BEHALF' (active subcase) or 'FORCE_CLOSE_INTERVENTION' (force-closed)

    Returns True if updated, False if subcase not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET DepartmentExplanationText    = ?,
                DepartmentEnteredByUserID    = ?,
                DepartmentEnteredForRole     = ?,
                DepartmentEntryMode          = ?,
                DepartmentEntryTimestamp     = ?,
                UpdatedAt                    = ?,
                UpdatedByUserID              = ?
            WHERE SubcaseID = ?
        """
        now = datetime.now()
        cursor.execute(query, (
            text,
            entered_by_user_id,
            entered_for_role,
            entry_mode,
            now,
            now,
            entered_by_user_id,
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def fill_administration_on_behalf(
    subcase_id: int,
    text: str,
    entered_by_user_id: int,
    entered_for_role: str,
    entry_mode: str
) -> bool:
    """
    Write administration explanation text with full ownership tracking.
    Used by COMPLAINT_SUPERVISOR / WORKER filling on behalf of ADMINISTRATION_ADMIN.

    entry_mode: 'ON_BEHALF' (active subcase) or 'FORCE_CLOSE_INTERVENTION' (force-closed)

    Returns True if updated, False if subcase not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET AdministrationExplanationText    = ?,
                AdministrationEnteredByUserID    = ?,
                AdministrationEnteredForRole     = ?,
                AdministrationEntryMode          = ?,
                AdministrationEntryTimestamp     = ?,
                UpdatedAt                        = ?,
                UpdatedByUserID                  = ?
            WHERE SubcaseID = ?
        """
        now = datetime.now()
        cursor.execute(query, (
            text,
            entered_by_user_id,
            entered_for_role,
            entry_mode,
            now,
            now,
            entered_by_user_id,
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def get_force_closed_subcases(status: str) -> List[Dict[str, Any]]:
    """
    Fetch force-closed subcases with full details for the Insight page tabs.

    Args:
        status: 'FORCE_CLOSED_DRAFT' or 'FORCE_CLOSED_COMPLETE'

    Returns:
        List of subcase dicts with case details, org unit info, and force-close metadata.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.CaseType,
                sub.Status,
                sub.CreatedAt,
                sub.ForceClosedAt,
                sub.ForceCloseReason,
                DATEDIFF(day, sub.CreatedAt, GETDATE()) AS WaitingDays,

                -- Org Unit Info
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,

                -- Incident Case Info
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,
                inc.incident_number AS IncidentNumber,
                CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
                CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,

                -- Seasonal Report Info
                sr.SeasonalReportID,
                s.SeasonName

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID

            WHERE sub.Status = ?
            ORDER BY sub.ForceClosedAt DESC
        """

        cursor.execute(query, (status,))
        rows = cursor.fetchall()

        result = []
        for row in rows:
            result.append({
                "subcase_id": row.SubcaseID,
                "case_type": row.CaseType,
                "status": row.Status,
                "created_at": str(row.CreatedAt) if row.CreatedAt else None,
                "force_closed_at": str(row.ForceClosedAt) if row.ForceClosedAt else None,
                "force_close_reason": row.ForceCloseReason,
                "waiting_days": row.WaitingDays,
                "target_org_unit_id": row.TargetOrgUnitID,
                "org_unit_name": row.OrgUnitName,
                "org_type": row.OrgType,
                "incident_request_case_id": row.IncidentRequestCaseID,
                "incident_number": row.IncidentNumber,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
                "is_red_flag": bool(row.IsRedFlag) if row.IsRedFlag is not None else False,
                "is_never_event": bool(row.IsNeverEvent) if row.IsNeverEvent is not None else False,
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName,
            })

        return result

    finally:
        cursor.close()
        conn.close()


def get_subcase_fill_state(subcase_id: int) -> Optional[Dict[str, Any]]:
    """
    Return the full manual-fill state for a subcase including ownership tracking
    for each level (Section, Department, Administration).

    Joins APP_Users three times to resolve entered_by usernames.

    Returns None if subcase not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            SELECT
                sub.SubcaseID,
                sub.Status,
                sub.IncidentRequestCaseID,
                sub.ForceCloseReason,
                sub.ForceClosedAt,

                -- Section
                sub.SectionExplanationText,
                sub.SectionEnteredForRole,
                sub.SectionEntryMode,
                sub.SectionEntryTimestamp,
                u_sec.Username   AS SectionEnteredByUsername,
                u_sec.DisplayName AS SectionEnteredByDisplayName,

                -- Department
                sub.DepartmentExplanationText,
                sub.DepartmentEnteredForRole,
                sub.DepartmentEntryMode,
                sub.DepartmentEntryTimestamp,
                u_dep.Username   AS DepartmentEnteredByUsername,
                u_dep.DisplayName AS DepartmentEnteredByDisplayName,

                -- Administration
                sub.AdministrationExplanationText,
                sub.AdministrationEnteredForRole,
                sub.AdministrationEntryMode,
                sub.AdministrationEntryTimestamp,
                u_adm.Username   AS AdministrationEnteredByUsername,
                u_adm.DisplayName AS AdministrationEnteredByDisplayName,

                -- Patient Services Decision
                sub.PatientServicesDecisionText,
                sub.PatientServicesDecisionAt,
                sub.PatientServicesDecisionUpdatedAt,
                u_ps.Username    AS PSDecisionByUsername,
                u_ps.DisplayName AS PSDecisionByDisplayName,

                -- Force-closed-by user
                u_fc.Username   AS ForceClosedByUsername,
                u_fc.DisplayName AS ForceClosedByDisplayName,

                -- Case context (for Manual Fill page display)
                ic.ComplaintText    AS CaseDescription,
                ic.PatientName,
                inc.incident_number AS IncidentNumber,
                dom.DomainName,
                cat.CategoryName,
                sc.SubCategoryName,
                cls.Classification_EN AS ClassificationEN,

                -- Originating section/unit (immutable on the parent case, unlike
                -- TargetOrgUnitID which can be repointed as the subcase escalates)
                issuing_org.Name AS IssuingOrgUnitName

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.APP_Users u_sec
                ON sub.SectionEnteredByUserID = u_sec.UserID
            LEFT JOIN dbo.APP_Users u_dep
                ON sub.DepartmentEnteredByUserID = u_dep.UserID
            LEFT JOIN dbo.APP_Users u_adm
                ON sub.AdministrationEnteredByUserID = u_adm.UserID
            LEFT JOIN dbo.APP_Users u_ps
                ON sub.PatientServicesDecisionByUserID = u_ps.UserID
            LEFT JOIN dbo.APP_Users u_fc
                ON sub.ForceClosedByUserID = u_fc.UserID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
            LEFT JOIN dbo.APP_LOOKUP_DOMAIN dom
                ON ic.DomainID = dom.DomainID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_LOOKUP_SUBCATEGORY sc
                ON ic.SubCategoryID = sc.SubCategoryID
            LEFT JOIN dbo.APP_LOOKUP_CLASSIFICATION cls
                ON ic.ClassificationID = cls.ClassificationID
            LEFT JOIN dbo.AdminsrationUnit issuing_org
                ON ic.IssuingOrgUnitID = issuing_org.UniqueID
            WHERE sub.SubcaseID = ?
        """
        cursor.execute(query, (subcase_id,))
        row = cursor.fetchone()
        if not row:
            return None

        def _username(uname, dname):
            return dname if dname else (uname if uname else None)

        def _ts(val):
            return val.isoformat() if val and hasattr(val, 'isoformat') else (str(val) if val else None)

        return {
            "subcase_id": row.SubcaseID,
            "status": row.Status,
            "incident_id": row.IncidentRequestCaseID,
            "force_close_reason": row.ForceCloseReason,
            "force_closed_at": _ts(row.ForceClosedAt),
            "force_closed_by": _username(row.ForceClosedByUsername, row.ForceClosedByDisplayName),
            "section": {
                "explanation_text": row.SectionExplanationText,
                "entered_by": _username(row.SectionEnteredByUsername, row.SectionEnteredByDisplayName),
                "entered_for_role": row.SectionEnteredForRole,
                "entry_mode": row.SectionEntryMode,
                "entry_timestamp": _ts(row.SectionEntryTimestamp),
            },
            "department": {
                "explanation_text": row.DepartmentExplanationText,
                "entered_by": _username(row.DepartmentEnteredByUsername, row.DepartmentEnteredByDisplayName),
                "entered_for_role": row.DepartmentEnteredForRole,
                "entry_mode": row.DepartmentEntryMode,
                "entry_timestamp": _ts(row.DepartmentEntryTimestamp),
            },
            "administration": {
                "explanation_text": row.AdministrationExplanationText,
                "entered_by": _username(row.AdministrationEnteredByUsername, row.AdministrationEnteredByDisplayName),
                "entered_for_role": row.AdministrationEnteredForRole,
                "entry_mode": row.AdministrationEntryMode,
                "entry_timestamp": _ts(row.AdministrationEntryTimestamp),
            },
            "patient_services_decision": {
                "decision_text": row.PatientServicesDecisionText,
                "entered_by": _username(row.PSDecisionByUsername, row.PSDecisionByDisplayName),
                "decision_at": _ts(row.PatientServicesDecisionAt),
                "updated_at": _ts(row.PatientServicesDecisionUpdatedAt),
            },
            "case_description": row.CaseDescription,
            "patient_name": row.PatientName,
            "incident_number": row.IncidentNumber,
            "domain_name": row.DomainName,
            "category_name": row.CategoryName,
            "sub_category_name": row.SubCategoryName,
            "classification_en": row.ClassificationEN,
            "issuing_org_unit_name": row.IssuingOrgUnitName,
        }
    finally:
        cursor.close()
        conn.close()


def get_supervisor_name_for_org_unit(
    org_unit_id: int,
    expected_role_code: Optional[str] = None
) -> Optional[str]:
    """
    Lookup supervisor/admin name for a given org unit.

    Queries APP_Users + APP_UserRoleScope to find the admin assigned to this unit.
    When expected_role_code is supplied (e.g. 'SECTION_ADMIN'), only users with
    that role are considered.  This prevents a DEPARTMENT_ADMIN who has downward
    scope visibility into child sections from being returned as the supervisor for
    a section-level org unit.

    Args:
        org_unit_id: Organizational unit ID
        expected_role_code: Optional role code to restrict the lookup (e.g.
            'SECTION_ADMIN', 'DEPARTMENT_ADMIN', 'ADMINISTRATION_ADMIN').
            When None, any active user linked to the unit is returned (legacy
            behaviour, kept as fallback).

    Returns:
        Supervisor name (DisplayName or Username) or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        if expected_role_code:
            query = """
                SELECT TOP 1
                    u.DisplayName,
                    u.Username
                FROM dbo.APP_Users u
                INNER JOIN dbo.APP_UserRoleScope urs
                    ON u.UserID = urs.UserID
                INNER JOIN dbo.APP_Roles r
                    ON urs.RoleID = r.RoleID
                WHERE urs.OrgUnitID = ?
                  AND u.IsActive = 1
                  AND r.RoleCode = ?
                ORDER BY u.UserID
            """
            cursor.execute(query, (org_unit_id, expected_role_code))
        else:
            query = """
                SELECT TOP 1
                    u.DisplayName,
                    u.Username
                FROM dbo.APP_Users u
                INNER JOIN dbo.APP_UserRoleScope urs
                    ON u.UserID = urs.UserID
                WHERE urs.OrgUnitID = ?
                  AND u.IsActive = 1
                ORDER BY u.UserID
            """
            cursor.execute(query, (org_unit_id,))

        row = cursor.fetchone()

        if row:
            # Prefer DisplayName, fallback to Username
            return row.DisplayName if row.DisplayName else row.Username

        return None

    finally:
        cursor.close()
        conn.close()


# ============================================================
# AUTOMATIC FORCE CLOSE (HCAT Automatic Force Close Policy - Session 4)
# ============================================================

def force_close_section(subcase_id: int, auto_text: str, system_user_id: int = 1) -> bool:
    """
    Mark a subcase Section-force-closed due to an expired SectionDeadlineAt.

    - Sets SectionForceClosedAt = now and SectionLateReply = 1.
    - Transitions Status to 'FORCE_CLOSED_AT_SECTION' (case remains visible
      and escalates to Department responsibility via the pending-for-department
      query).
    - If SectionExplanationText is empty, fills it with auto_text and tags the
      entry with SectionEntryMode = 'AUTO_FORCE_CLOSE' (so it can be identified
      and replaced later by Give More Time). A real existing answer is never
      overwritten.
    - Idempotent: WHERE SectionForceClosedAt IS NULL guard means re-running
      this on an already force-closed subcase is a no-op.

    Returns:
        True if a row was updated, False if not found or already force-closed.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET SectionForceClosedAt = ?,
                SectionLateReply      = 1,
                Status                = 'FORCE_CLOSED_AT_SECTION',
                SectionExplanationText = CASE
                    WHEN SectionExplanationText IS NULL OR LTRIM(RTRIM(SectionExplanationText)) = ''
                    THEN ? ELSE SectionExplanationText END,
                SectionEnteredByUserID = CASE
                    WHEN SectionExplanationText IS NULL OR LTRIM(RTRIM(SectionExplanationText)) = ''
                    THEN ? ELSE SectionEnteredByUserID END,
                SectionEnteredForRole = CASE
                    WHEN SectionExplanationText IS NULL OR LTRIM(RTRIM(SectionExplanationText)) = ''
                    THEN ? ELSE SectionEnteredForRole END,
                SectionEntryMode = CASE
                    WHEN SectionExplanationText IS NULL OR LTRIM(RTRIM(SectionExplanationText)) = ''
                    THEN ? ELSE SectionEntryMode END,
                SectionEntryTimestamp = CASE
                    WHEN SectionExplanationText IS NULL OR LTRIM(RTRIM(SectionExplanationText)) = ''
                    THEN ? ELSE SectionEntryTimestamp END,
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND SectionForceClosedAt IS NULL
        """
        cursor.execute(query, (
            now,                # SectionForceClosedAt
            auto_text,          # SectionExplanationText (if empty)
            system_user_id,     # SectionEnteredByUserID (if empty)
            'SECTION_ADMIN',    # SectionEnteredForRole (if empty)
            'AUTO_FORCE_CLOSE', # SectionEntryMode (if empty)
            now,                # SectionEntryTimestamp (if empty)
            now,                # UpdatedAt
            system_user_id,     # UpdatedByUserID
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def force_close_department(subcase_id: int, auto_text: str, system_user_id: int = 1) -> bool:
    """
    Mark a subcase Department-force-closed due to an expired DepartmentDeadlineAt.

    - Sets DepartmentForceClosedAt = now and DepartmentLateReply = 1.
    - Transitions Status to 'FORCE_CLOSED_AT_DEPARTMENT' (case remains visible
      and escalates to Administration responsibility via the
      pending-for-administration query).
    - If DepartmentExplanationText is empty, fills it with auto_text and tags
      the entry with DepartmentEntryMode = 'AUTO_FORCE_CLOSE'. A real existing
      answer is never overwritten.
    - Idempotent: WHERE DepartmentForceClosedAt IS NULL guard.

    Returns:
        True if a row was updated, False if not found or already force-closed.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET DepartmentForceClosedAt = ?,
                DepartmentLateReply      = 1,
                Status                   = 'FORCE_CLOSED_AT_DEPARTMENT',
                DepartmentExplanationText = CASE
                    WHEN DepartmentExplanationText IS NULL OR LTRIM(RTRIM(DepartmentExplanationText)) = ''
                    THEN ? ELSE DepartmentExplanationText END,
                DepartmentEnteredByUserID = CASE
                    WHEN DepartmentExplanationText IS NULL OR LTRIM(RTRIM(DepartmentExplanationText)) = ''
                    THEN ? ELSE DepartmentEnteredByUserID END,
                DepartmentEnteredForRole = CASE
                    WHEN DepartmentExplanationText IS NULL OR LTRIM(RTRIM(DepartmentExplanationText)) = ''
                    THEN ? ELSE DepartmentEnteredForRole END,
                DepartmentEntryMode = CASE
                    WHEN DepartmentExplanationText IS NULL OR LTRIM(RTRIM(DepartmentExplanationText)) = ''
                    THEN ? ELSE DepartmentEntryMode END,
                DepartmentEntryTimestamp = CASE
                    WHEN DepartmentExplanationText IS NULL OR LTRIM(RTRIM(DepartmentExplanationText)) = ''
                    THEN ? ELSE DepartmentEntryTimestamp END,
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND DepartmentForceClosedAt IS NULL
        """
        cursor.execute(query, (
            now,                   # DepartmentForceClosedAt
            auto_text,             # DepartmentExplanationText (if empty)
            system_user_id,        # DepartmentEnteredByUserID (if empty)
            'DEPARTMENT_ADMIN',    # DepartmentEnteredForRole (if empty)
            'AUTO_FORCE_CLOSE',    # DepartmentEntryMode (if empty)
            now,                   # DepartmentEntryTimestamp (if empty)
            now,                   # UpdatedAt
            system_user_id,        # UpdatedByUserID
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def force_close_administration(subcase_id: int, auto_text: str, system_user_id: int = 1) -> bool:
    """
    Mark a subcase Administration-force-closed due to an expired
    AdministrationDeadlineAt.

    - Sets AdministrationForceClosedAt = now and AdministrationLateReply = 1.
    - Transitions Status to 'FORCE_CLOSED_AT_ADMINISTRATION' (final status -
      case remains visible for Administration/Supervisor follow-up, it is
      never deleted or hidden).
    - If AdministrationExplanationText is empty, fills it with auto_text and
      tags the entry with AdministrationEntryMode = 'AUTO_FORCE_CLOSE'. A real
      existing answer is never overwritten.
    - Idempotent: WHERE AdministrationForceClosedAt IS NULL guard.

    Returns:
        True if a row was updated, False if not found or already force-closed.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET AdministrationForceClosedAt = ?,
                AdministrationLateReply      = 1,
                Status                       = 'FORCE_CLOSED_AT_ADMINISTRATION',
                AdministrationExplanationText = CASE
                    WHEN AdministrationExplanationText IS NULL OR LTRIM(RTRIM(AdministrationExplanationText)) = ''
                    THEN ? ELSE AdministrationExplanationText END,
                AdministrationEnteredByUserID = CASE
                    WHEN AdministrationExplanationText IS NULL OR LTRIM(RTRIM(AdministrationExplanationText)) = ''
                    THEN ? ELSE AdministrationEnteredByUserID END,
                AdministrationEnteredForRole = CASE
                    WHEN AdministrationExplanationText IS NULL OR LTRIM(RTRIM(AdministrationExplanationText)) = ''
                    THEN ? ELSE AdministrationEnteredForRole END,
                AdministrationEntryMode = CASE
                    WHEN AdministrationExplanationText IS NULL OR LTRIM(RTRIM(AdministrationExplanationText)) = ''
                    THEN ? ELSE AdministrationEntryMode END,
                AdministrationEntryTimestamp = CASE
                    WHEN AdministrationExplanationText IS NULL OR LTRIM(RTRIM(AdministrationExplanationText)) = ''
                    THEN ? ELSE AdministrationEntryTimestamp END,
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND AdministrationForceClosedAt IS NULL
        """
        cursor.execute(query, (
            now,                      # AdministrationForceClosedAt
            auto_text,                # AdministrationExplanationText (if empty)
            system_user_id,           # AdministrationEnteredByUserID (if empty)
            'ADMINISTRATION_ADMIN',   # AdministrationEnteredForRole (if empty)
            'AUTO_FORCE_CLOSE',       # AdministrationEntryMode (if empty)
            now,                      # AdministrationEntryTimestamp (if empty)
            now,                      # UpdatedAt
            system_user_id,           # UpdatedByUserID
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def get_overdue_subcases_for_level(deadline_column: str, pending_statuses: List[str]) -> List[Dict[str, Any]]:
    """
    Find subcases whose <Level>DeadlineAt has expired and whose Status is
    still one of the "pending" statuses for that level (i.e. the case has not
    yet been force-closed at this level).

    Joins to APP_IncidentCase to expose RecordTypeID and IsMorbidity so the
    service layer can exclude Notice (RecordTypeID=2) and Morbidity
    (IsMorbidity=1) cases. Seasonal-report subcases have no
    IncidentRequestCaseID, so the join is a LEFT JOIN and RecordTypeID /
    IsMorbidity come back NULL for them (treated as "not Notice, not
    Morbidity" by the service layer).

    Args:
        deadline_column: 'SectionDeadlineAt', 'DepartmentDeadlineAt' or
            'AdministrationDeadlineAt'
        pending_statuses: Status values that mean "this level is still
            responsible / pending" for the given deadline column.

    Returns:
        List of dicts with SubcaseID, RecordTypeID, IsMorbidity.
    """
    if deadline_column not in (
        "SectionDeadlineAt", "DepartmentDeadlineAt", "AdministrationDeadlineAt"
    ):
        raise ValueError(f"Unsupported deadline column: {deadline_column}")

    conn = get_connection()
    cursor = conn.cursor()
    try:
        placeholders = ", ".join("?" for _ in pending_statuses)
        query = f"""
            SELECT s.SubcaseID, c.RecordTypeID, c.IsMorbidity
            FROM dbo.APP_AdministrativeSubcase s
            LEFT JOIN dbo.APP_IncidentCase c
                ON s.IncidentRequestCaseID = c.IncidentRequestCaseID
            WHERE s.{deadline_column} IS NOT NULL
              AND s.{deadline_column} < ?
              AND s.Status IN ({placeholders})
        """
        cursor.execute(query, (datetime.now(), *pending_statuses))
        rows = cursor.fetchall()
        return [
            {
                "SubcaseID": row.SubcaseID,
                "RecordTypeID": row.RecordTypeID,
                "IsMorbidity": row.IsMorbidity,
            }
            for row in rows
        ]
    finally:
        cursor.close()
        conn.close()


# ============================================================
# GIVE MORE TIME WORKFLOW (HCAT Automatic Force Close Policy - Session 5)
# ============================================================

def get_subcase_deadline_state(subcase_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch per-level deadline / force-close / late-reply / extra-time-grant
    state for a subcase.

    Used by the Give More Time workflow to validate the current workflow
    level before restoration and to return the updated workflow state to the
    caller afterwards.

    Returns:
        Dict with subcase_id, status, target_org_unit_id and, for each of
        section/department/administration: deadline_at, force_closed_at,
        late_reply, extra_time_granted_at, extra_time_granted_by.
        None if the subcase does not exist.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            SELECT
                SubcaseID, Status, TargetOrgUnitID,
                SectionDeadlineAt, SectionForceClosedAt, SectionLateReply,
                SectionExtraTimeGrantedAt, SectionExtraTimeGrantedBy,
                DepartmentDeadlineAt, DepartmentForceClosedAt, DepartmentLateReply,
                DepartmentExtraTimeGrantedAt, DepartmentExtraTimeGrantedBy,
                AdministrationDeadlineAt, AdministrationForceClosedAt, AdministrationLateReply,
                AdministrationExtraTimeGrantedAt, AdministrationExtraTimeGrantedBy
            FROM dbo.APP_AdministrativeSubcase
            WHERE SubcaseID = ?
        """
        cursor.execute(query, (subcase_id,))
        row = cursor.fetchone()
        if not row:
            return None

        return {
            "subcase_id": row.SubcaseID,
            "status": row.Status,
            "target_org_unit_id": row.TargetOrgUnitID,
            "section": {
                "deadline_at": row.SectionDeadlineAt,
                "force_closed_at": row.SectionForceClosedAt,
                "late_reply": bool(row.SectionLateReply),
                "extra_time_granted_at": row.SectionExtraTimeGrantedAt,
                "extra_time_granted_by": row.SectionExtraTimeGrantedBy,
            },
            "department": {
                "deadline_at": row.DepartmentDeadlineAt,
                "force_closed_at": row.DepartmentForceClosedAt,
                "late_reply": bool(row.DepartmentLateReply),
                "extra_time_granted_at": row.DepartmentExtraTimeGrantedAt,
                "extra_time_granted_by": row.DepartmentExtraTimeGrantedBy,
            },
            "administration": {
                "deadline_at": row.AdministrationDeadlineAt,
                "force_closed_at": row.AdministrationForceClosedAt,
                "late_reply": bool(row.AdministrationLateReply),
                "extra_time_granted_at": row.AdministrationExtraTimeGrantedAt,
                "extra_time_granted_by": row.AdministrationExtraTimeGrantedBy,
            },
        }
    finally:
        cursor.close()
        conn.close()


def give_section_more_time(subcase_id: int, granted_by_user_id: int, deadline_days: int) -> bool:
    """
    Restore a Section-force-closed subcase to active Section responsibility
    (HCAT Give More Time workflow - Session 5).

    - Sets SectionDeadlineAt = now + deadline_days (deadline restarts from now).
    - Sets SectionExtraTimeGrantedAt = now and SectionExtraTimeGrantedBy =
      granted_by_user_id (history of who granted the extension).
    - Transitions Status back to 'SUBMITTED_TO_SECTION' (Section's normal
      pending status) - Section becomes responsible/editable again.
    - SectionForceClosedAt and SectionLateReply are left UNCHANGED - the
      force-close history and late-reply flag are preserved permanently.
    - Guard: WHERE Status = 'FORCE_CLOSED_AT_SECTION' prevents restoring a
      subcase that is not currently force-closed at Section level and
      prevents double-restoration (status no longer matches after the first
      restore).

    Returns:
        True if a row was updated, False if the subcase is not currently
        FORCE_CLOSED_AT_SECTION (not found, wrong level, or already restored).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        new_deadline = now + timedelta(days=deadline_days)
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET SectionDeadlineAt        = ?,
                SectionExtraTimeGrantedAt = ?,
                SectionExtraTimeGrantedBy = ?,
                Status                   = 'SUBMITTED_TO_SECTION',
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND Status = 'FORCE_CLOSED_AT_SECTION'
        """
        cursor.execute(query, (
            new_deadline,        # SectionDeadlineAt
            now,                 # SectionExtraTimeGrantedAt
            granted_by_user_id,  # SectionExtraTimeGrantedBy
            now,                 # UpdatedAt
            granted_by_user_id,  # UpdatedByUserID
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def give_department_more_time(subcase_id: int, granted_by_user_id: int, deadline_days: int) -> bool:
    """
    Restore a Department-force-closed subcase to active Department
    responsibility (HCAT Give More Time workflow - Session 5).

    - Sets DepartmentDeadlineAt = now + deadline_days (deadline restarts from now).
    - Sets DepartmentExtraTimeGrantedAt = now and DepartmentExtraTimeGrantedBy =
      granted_by_user_id.
    - Transitions Status back to 'SECTION_ACCEPTED_PENDING_DEPT' (Department's
      normal pending status) - Department becomes responsible/editable again.
    - DepartmentForceClosedAt and DepartmentLateReply are left UNCHANGED -
      force-close history and late-reply flag are preserved permanently.
    - Guard: WHERE Status = 'FORCE_CLOSED_AT_DEPARTMENT' prevents restoring a
      subcase not currently force-closed at Department level and prevents
      double-restoration.

    Returns:
        True if a row was updated, False if the subcase is not currently
        FORCE_CLOSED_AT_DEPARTMENT (not found, wrong level, or already restored).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        new_deadline = now + timedelta(days=deadline_days)
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET DepartmentDeadlineAt        = ?,
                DepartmentExtraTimeGrantedAt = ?,
                DepartmentExtraTimeGrantedBy = ?,
                Status                      = 'SECTION_ACCEPTED_PENDING_DEPT',
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND Status = 'FORCE_CLOSED_AT_DEPARTMENT'
        """
        cursor.execute(query, (
            new_deadline,        # DepartmentDeadlineAt
            now,                 # DepartmentExtraTimeGrantedAt
            granted_by_user_id,  # DepartmentExtraTimeGrantedBy
            now,                 # UpdatedAt
            granted_by_user_id,  # UpdatedByUserID
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


def give_administration_more_time(subcase_id: int, granted_by_user_id: int, deadline_days: int) -> bool:
    """
    Restore an Administration-force-closed subcase to active Administration
    responsibility (HCAT Give More Time workflow - Session 5).

    - Sets AdministrationDeadlineAt = now + deadline_days (deadline restarts
      from now).
    - Sets AdministrationExtraTimeGrantedAt = now and
      AdministrationExtraTimeGrantedBy = granted_by_user_id.
    - Transitions Status back to 'DEPT_ACCEPTED_PENDING_ADMIN'
      (Administration's normal pending status) - Administration becomes
      responsible/editable again.
    - AdministrationForceClosedAt and AdministrationLateReply are left
      UNCHANGED - force-close history and late-reply flag are preserved
      permanently.
    - Guard: WHERE Status = 'FORCE_CLOSED_AT_ADMINISTRATION' prevents
      restoring a subcase not currently force-closed at Administration level
      and prevents double-restoration.

    Returns:
        True if a row was updated, False if the subcase is not currently
        FORCE_CLOSED_AT_ADMINISTRATION (not found, wrong level, or already
        restored).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        now = datetime.now()
        new_deadline = now + timedelta(days=deadline_days)
        query = """
            UPDATE dbo.APP_AdministrativeSubcase
            SET AdministrationDeadlineAt        = ?,
                AdministrationExtraTimeGrantedAt = ?,
                AdministrationExtraTimeGrantedBy = ?,
                Status                          = 'DEPT_ACCEPTED_PENDING_ADMIN',
                UpdatedAt       = ?,
                UpdatedByUserID = ?
            WHERE SubcaseID = ?
              AND Status = 'FORCE_CLOSED_AT_ADMINISTRATION'
        """
        cursor.execute(query, (
            new_deadline,        # AdministrationDeadlineAt
            now,                 # AdministrationExtraTimeGrantedAt
            granted_by_user_id,  # AdministrationExtraTimeGrantedBy
            now,                 # UpdatedAt
            granted_by_user_id,  # UpdatedByUserID
            subcase_id
        ))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        cursor.close()
        conn.close()


# ============================================================
# INVESTIGATION HISTORY (Stage 6 – read-only)
# ============================================================

def get_subcase_history(subcase_id: int) -> Optional[Dict[str, Any]]:
    """
    Return all investigation-level texts for a subcase, plus available
    per-level metadata (entered_by, timestamp from on-behalf fills).

    Used by the frontend InvestigationHistorySection to display the full
    response history without any workflow logic.

    Returns None if the subcase is not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        query = """
            SELECT
                sub.SubcaseID,
                org.Name AS OrgUnitName,

                sub.SectionExplanationText,
                sub.SectionEntryTimestamp,
                u_sec.DisplayName AS SectionEnteredByName,

                sub.DepartmentExplanationText,
                sub.DepartmentEntryTimestamp,
                u_dep.DisplayName AS DepartmentEnteredByName,

                sub.AdministrationExplanationText,
                sub.AdministrationEntryTimestamp,
                u_adm.DisplayName AS AdministrationEnteredByName,

                sub.PatientServicesDecisionText,
                sub.PatientServicesDecisionAt,
                u_ps.DisplayName  AS PSDecisionByName

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_Users u_sec
                ON sub.SectionEnteredByUserID = u_sec.UserID
            LEFT JOIN dbo.APP_Users u_dep
                ON sub.DepartmentEnteredByUserID = u_dep.UserID
            LEFT JOIN dbo.APP_Users u_adm
                ON sub.AdministrationEnteredByUserID = u_adm.UserID
            LEFT JOIN dbo.APP_Users u_ps
                ON sub.PatientServicesDecisionByUserID = u_ps.UserID
            WHERE sub.SubcaseID = ?
        """
        cursor.execute(query, (subcase_id,))
        row = cursor.fetchone()
        if not row:
            return None

        def _ts(val):
            return val.isoformat() if val and hasattr(val, 'isoformat') else None

        return {
            "subcase_id": subcase_id,
            "org_unit_name": row.OrgUnitName,
            "section": {
                "has_content": bool(row.SectionExplanationText),
                "text": row.SectionExplanationText,
                "entered_by": row.SectionEnteredByName,
                "entered_at": _ts(row.SectionEntryTimestamp),
            },
            "department": {
                "has_content": bool(row.DepartmentExplanationText),
                "text": row.DepartmentExplanationText,
                "entered_by": row.DepartmentEnteredByName,
                "entered_at": _ts(row.DepartmentEntryTimestamp),
            },
            "administration": {
                "has_content": bool(row.AdministrationExplanationText),
                "text": row.AdministrationExplanationText,
                "entered_by": row.AdministrationEnteredByName,
                "entered_at": _ts(row.AdministrationEntryTimestamp),
            },
            "patient_services": {
                "has_content": bool(row.PatientServicesDecisionText),
                "text": row.PatientServicesDecisionText,
                "entered_by": row.PSDecisionByName,
                "entered_at": _ts(row.PatientServicesDecisionAt),
            },
        }
    finally:
        cursor.close()
        conn.close()


# ============================================================
# ACCOUNTABILITY QUERIES  (HCAT Force-Close Accountability Box)
# ============================================================

_ACCOUNTABILITY_SELECT = """
    SELECT
        sub.SubcaseID,
        sub.CaseType,
        sub.IncidentRequestCaseID,
        sub.SeasonalReportID,
        sub.TargetOrgUnitID,
        org.Name  AS OrgUnitName,
        org.Type  AS OrgUnitType,
        sub.Status,
        sub.SectionExplanationText,
        sub.SectionRejectionText,
        sub.DepartmentExplanationText,
        sub.DepartmentRejectionText,
        sub.AdministrationExplanationText,
        sub.AdministrationRejectionText,
        sub.CreatedAt,
        sub.CreatedByUserID,
        sub.UpdatedAt,
        sub.UpdatedByUserID,
        inc.incident_number AS IncidentNumber,
        sub.SectionDeadlineAt,
        sub.DepartmentDeadlineAt,
        sub.AdministrationDeadlineAt,
        sub.SectionForceClosedAt,
        sub.SectionLateReply,
        sub.SectionExtraTimeGrantedAt,
        sub.DepartmentForceClosedAt,
        sub.DepartmentLateReply,
        sub.DepartmentExtraTimeGrantedAt,
        sub.AdministrationForceClosedAt,
        sub.AdministrationLateReply,
        sub.AdministrationExtraTimeGrantedAt,
        CASE WHEN ic.ClinicalRiskTypeID = 2 THEN 1 ELSE 0 END AS IsRedFlag,
        CASE WHEN ic.ClinicalRiskTypeID = 3 THEN 1 ELSE 0 END AS IsNeverEvent,
        ISNULL(ic.IsMorbidity, 0) AS IsMorbidity,
        ic.FeedbackRecievedDate,
        ic.IncidentDate,
        ic.RecordTypeID
    FROM dbo.APP_AdministrativeSubcase sub
    LEFT JOIN dbo.AdminsrationUnit org
        ON sub.TargetOrgUnitID = org.UniqueID
    LEFT JOIN dbo.APP_IncidentCase ic
        ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
    LEFT JOIN dbo.APP_Incident inc
        ON ic.incident_id = inc.incident_id
"""


def _rows_to_subcase_dicts(rows) -> List[Dict[str, Any]]:
    result = []
    for row in rows:
        result.append({
            "subcase_id": row.SubcaseID,
            "case_type": row.CaseType,
            "incident_request_case_id": row.IncidentRequestCaseID,
            "seasonal_report_id": row.SeasonalReportID,
            "target_org_unit_id": row.TargetOrgUnitID,
            "org_unit_name": row.OrgUnitName,
            "target_org_unit_type": row.OrgUnitType,
            "status": row.Status,
            "section_explanation_text": row.SectionExplanationText,
            "section_rejection_text": row.SectionRejectionText,
            "department_explanation_text": row.DepartmentExplanationText,
            "department_rejection_text": row.DepartmentRejectionText,
            "administration_explanation_text": row.AdministrationExplanationText,
            "administration_rejection_text": row.AdministrationRejectionText,
            "created_at": row.CreatedAt,
            "created_by_user_id": row.CreatedByUserID,
            "updated_at": row.UpdatedAt,
            "updated_by_user_id": row.UpdatedByUserID,
            "incident_number": row.IncidentNumber,
            "section_deadline_at": row.SectionDeadlineAt,
            "department_deadline_at": row.DepartmentDeadlineAt,
            "administration_deadline_at": row.AdministrationDeadlineAt,
            "section_force_closed_at": row.SectionForceClosedAt,
            "section_late_reply": bool(row.SectionLateReply),
            "section_extra_time_granted_at": row.SectionExtraTimeGrantedAt,
            "department_force_closed_at": row.DepartmentForceClosedAt,
            "department_late_reply": bool(row.DepartmentLateReply),
            "department_extra_time_granted_at": row.DepartmentExtraTimeGrantedAt,
            "administration_force_closed_at": row.AdministrationForceClosedAt,
            "administration_late_reply": bool(row.AdministrationLateReply),
            "administration_extra_time_granted_at": row.AdministrationExtraTimeGrantedAt,
            "is_red_flag": bool(row.IsRedFlag),
            "is_never_event": bool(row.IsNeverEvent),
            "is_morbidity": bool(row.IsMorbidity),
            "feedback_received_date": row.FeedbackRecievedDate,
            "incident_date": row.IncidentDate,
            "record_type_id": row.RecordTypeID,
        })
    return result


def get_section_accountability_red() -> List[Dict[str, Any]]:
    """RED: force-closed at Section, not yet given more time."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            _ACCOUNTABILITY_SELECT +
            " WHERE sub.Status = 'FORCE_CLOSED_AT_SECTION'"
            "   AND sub.SectionExtraTimeGrantedAt IS NULL"
            " ORDER BY sub.SectionForceClosedAt DESC"
        )
        return _rows_to_subcase_dicts(cursor.fetchall())
    finally:
        cursor.close()
        conn.close()


def get_section_accountability_gray() -> List[Dict[str, Any]]:
    """GRAY: was force-closed at Section, given more time, progressed past Section."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            _ACCOUNTABILITY_SELECT +
            " WHERE sub.SectionForceClosedAt IS NOT NULL"
            "   AND sub.SectionExtraTimeGrantedAt IS NOT NULL"
            "   AND sub.Status NOT IN ("
            "       'SUBMITTED_TO_SECTION','RETURNED_TO_SECTION_FOR_REVISION',"
            "       'FORCE_CLOSED_AT_SECTION'"
            "   )"
            " ORDER BY sub.SectionExtraTimeGrantedAt DESC"
        )
        return _rows_to_subcase_dicts(cursor.fetchall())
    finally:
        cursor.close()
        conn.close()


def get_department_accountability_red() -> List[Dict[str, Any]]:
    """RED: force-closed at Department, not yet given more time."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            _ACCOUNTABILITY_SELECT +
            " WHERE sub.Status = 'FORCE_CLOSED_AT_DEPARTMENT'"
            "   AND sub.DepartmentExtraTimeGrantedAt IS NULL"
            " ORDER BY sub.DepartmentForceClosedAt DESC"
        )
        return _rows_to_subcase_dicts(cursor.fetchall())
    finally:
        cursor.close()
        conn.close()


def get_department_accountability_gray() -> List[Dict[str, Any]]:
    """GRAY: was force-closed at Department, given more time, progressed past Department."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            _ACCOUNTABILITY_SELECT +
            " WHERE sub.DepartmentForceClosedAt IS NOT NULL"
            "   AND sub.DepartmentExtraTimeGrantedAt IS NOT NULL"
            "   AND sub.Status NOT IN ("
            "       'SECTION_ACCEPTED_PENDING_DEPT','RETURNED_TO_DEPT_FOR_REVISION',"
            "       'FORCE_CLOSED_AT_DEPARTMENT'"
            "   )"
            " ORDER BY sub.DepartmentExtraTimeGrantedAt DESC"
        )
        return _rows_to_subcase_dicts(cursor.fetchall())
    finally:
        cursor.close()
        conn.close()
