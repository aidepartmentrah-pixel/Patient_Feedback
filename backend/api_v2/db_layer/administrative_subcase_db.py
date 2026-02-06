"""
Administrative Subcase Database Layer (API V2)
Handles SQL operations for APP_AdministrativeSubcase table.

This is part of Phase 3 parallel workflow system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import pyodbc


def get_db_connection():
    """Get database connection using project standard."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
            "updated_by_user_id": row.UpdatedByUserID
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
            WHERE Status = ?
            ORDER BY CreatedAt DESC
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
    conn = get_db_connection()
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
