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
                sub.UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
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
                "updated_by_user_id": row.UpdatedByUserID
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
    Should be called when transitioning to FORCE_CLOSED status.
    
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
    force_close_reason: str
) -> bool:
    """
    Force close a subcase with full tracking (status + tracking fields).
    
    This is a convenience function that:
    1. Updates status to FORCE_CLOSED
    2. Sets force close tracking fields
    
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
            SET Status = 'FORCE_CLOSED',
                ForceClosedAt = ?,
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
                
                -- Org Unit Info
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,
                
                -- Incident Case Info (for INCIDENT_RESPONSE)
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,
                
                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName
                
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org 
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic 
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev 
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat 
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr 
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID
            
            WHERE sub.Status IN ('SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION')
              AND sub.Status != 'FORCE_CLOSED'
            
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
                "incident_request_case_id": row.IncidentRequestCaseID,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
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
    
    Same structure as get_subcases_with_details_for_section but filters for:
    - SECTION_ACCEPTED_PENDING_DEPT
    - RETURNED_TO_DEPT_FOR_REVISION
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
                
                -- Org Unit Info
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,
                
                -- Incident Case Info (for INCIDENT_RESPONSE)
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,
                
                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName
                
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org 
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic 
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev 
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat 
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr 
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID
            
            WHERE sub.Status IN ('SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION')
              AND sub.Status != 'FORCE_CLOSED'
            
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
                "incident_request_case_id": row.IncidentRequestCaseID,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
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
    
    Same structure as get_subcases_with_details_for_section but filters for:
    - DEPT_ACCEPTED_PENDING_ADMIN
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
                
                -- Org Unit Info
                org.UniqueID AS TargetOrgUnitID,
                org.Name AS OrgUnitName,
                org.Type AS OrgType,
                
                -- Incident Case Info (for INCIDENT_RESPONSE)
                ic.IncidentRequestCaseID,
                ic.ComplaintText AS CaseDescription,
                ic.PatientName,
                ic.SeverityID,
                sev.SeverityName,
                cat.CategoryName,
                
                -- Seasonal Report Info (for SEASONAL_REPORT_RESPONSE)
                sr.SeasonalReportID,
                s.SeasonName
                
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org 
                ON sub.TargetOrgUnitID = org.UniqueID
            LEFT JOIN dbo.APP_IncidentCase ic 
                ON sub.IncidentRequestCaseID = ic.IncidentRequestCaseID
            LEFT JOIN dbo.APP_LOOKUP_SEVERITY sev 
                ON ic.SeverityID = sev.SeverityID
            LEFT JOIN dbo.APP_LOOKUP_CATEGORY cat 
                ON ic.CategoryID = cat.CategoryID
            LEFT JOIN dbo.APP_SeasonalOrgUnitReport sr 
                ON sub.SeasonalReportID = sr.SeasonalReportID
            LEFT JOIN dbo.Season s
                ON sr.SeasonID = s.UniqueID
            
            WHERE sub.Status = 'DEPT_ACCEPTED_PENDING_ADMIN'
              AND sub.Status != 'FORCE_CLOSED'
            
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
                "incident_request_case_id": row.IncidentRequestCaseID,
                "case_description": row.CaseDescription,
                "patient_name": row.PatientName,
                "severity_id": row.SeverityID,
                "severity": row.SeverityName,
                "category": row.CategoryName,
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
                sub.UpdatedByUserID
            FROM dbo.APP_AdministrativeSubcase sub
            LEFT JOIN dbo.AdminsrationUnit org
                ON sub.TargetOrgUnitID = org.UniqueID
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
                "updated_by_user_id": row.UpdatedByUserID
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
    - FORCE_CLOSED: Forcibly closed (terminal)
    
    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "SECTION_ACCEPTED_PENDING_DEPT",
        "RETURNED_TO_DEPT_FOR_REVISION",
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED",
        "SECTION_DENIED",
        "FORCE_CLOSED"
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
    - FORCE_CLOSED: Forcibly closed (terminal)
    
    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "DEPT_ACCEPTED_PENDING_ADMIN",
        "ADMIN_APPROVED",
        "RETURNED_TO_SECTION_FOR_REVISION",
        "FORCE_CLOSED"
    ]
    return get_subcases_by_statuses(archive_statuses)


def get_subcases_archived_for_administration() -> List[Dict[str, Any]]:
    """
    Fetch subcases that have moved past the administration stage.
    
    These are cases the admin processed (approved or force-closed).
    Statuses: Terminal statuses:
    - ADMIN_APPROVED: Final approval (workflow complete)
    - FORCE_CLOSED: Forcibly closed (terminal)
    - RETURNED_TO_DEPT_FOR_REVISION: Admin sent back to department (still admin processed it)
    
    Returns:
        List of subcase dicts ordered by UpdatedAt DESC
    """
    archive_statuses = [
        "ADMIN_APPROVED",
        "FORCE_CLOSED",
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


def get_supervisor_name_for_org_unit(org_unit_id: int) -> Optional[str]:
    """
    Lookup supervisor/admin name for a given org unit.
    
    Queries APP_Users + APP_UserRoleScope to find admin assigned to this unit.
    Returns DisplayName or Username.
    Returns None if no admin assigned.
    
    Args:
        org_unit_id: Organizational unit ID
    
    Returns:
        Supervisor name (DisplayName or Username) or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
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
