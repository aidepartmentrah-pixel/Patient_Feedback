"""
Administrative Subcase Database Layer (API V2)
Handles SQL operations for APP_AdministrativeSubcase table.

This is part of Phase 3 parallel workflow system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
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
    initial_status: str = "SUBMITTED_TO_SECTION"
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
                CreatedByUserID
            )
            OUTPUT INSERTED.SubcaseID
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(query, (
            case_type,
            incident_id,
            seasonal_report_id,
            target_org_unit_id,
            initial_status,
            datetime.now(),
            created_by_user_id
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
                inc.incident_number AS IncidentNumber
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
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
    
    Returns:
        List of subcase dicts
    """
    # Get both initial submissions and returned-for-revision cases
    initial = get_subcases_by_status("SECTION_ACCEPTED_PENDING_DEPT")
    returned = get_subcases_by_status("RETURNED_TO_DEPT_FOR_REVISION")
    return initial + returned


def get_subcases_pending_for_administration() -> List[Dict[str, Any]]:
    """
    Fetch subcases pending administration response.
    Status = 'DEPT_ACCEPTED_PENDING_ADMIN'
    
    Returns:
        List of subcase dicts
    """
    return get_subcases_by_status("DEPT_ACCEPTED_PENDING_ADMIN")


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

                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
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
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName
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

                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName

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
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName
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

                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName

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
                "seasonal_report_id": row.SeasonalReportID,
                "season_name": row.SeasonName
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
                inc.incident_number AS IncidentNumber
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_Incident inc
                ON ic.incident_id = inc.incident_id
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
        "FORCE_CLOSED_COMPLETE"
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

    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED",
        "RETURNED_TO_SECTION_FOR_REVISION",
        "FORCE_CLOSED",
        "FORCE_CLOSED_DRAFT",
        "FORCE_CLOSED_COMPLETE"
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

    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "ADMIN_APPROVED",
        "FORCE_CLOSED",
        "FORCE_CLOSED_DRAFT",
        "FORCE_CLOSED_COMPLETE",
        "RETURNED_TO_DEPT_FOR_REVISION"
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
    
    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "RETURNED_TO_SECTION_FOR_REVISION",
        "SECTION_ACCEPTED_PENDING_DEPT",
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED"
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

                -- Force-closed-by user
                u_fc.Username   AS ForceClosedByUsername,
                u_fc.DisplayName AS ForceClosedByDisplayName

            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.APP_Users u_sec
                ON sub.SectionEnteredByUserID = u_sec.UserID
            LEFT JOIN dbo.APP_Users u_dep
                ON sub.DepartmentEnteredByUserID = u_dep.UserID
            LEFT JOIN dbo.APP_Users u_adm
                ON sub.AdministrationEnteredByUserID = u_adm.UserID
            LEFT JOIN dbo.APP_Users u_fc
                ON sub.ForceClosedByUserID = u_fc.UserID
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
