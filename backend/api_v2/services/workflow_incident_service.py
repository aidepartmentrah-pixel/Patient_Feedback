"""
Workflow Incident Service — API v2

Provides read-only incident detail views for the workflow inbox.
Section admins, department admins, and administration admins can view
incident details without needing full edit permissions.

Authorization:
- User must have an active subcase assigned to their org unit for this incident
- OR user must be COMPLAINT_SUPERVISOR or higher (SOFTWARE_ADMIN)

This service is READ-ONLY:
- No writes, no updates, no deletes
- No workflow transitions
- No side effects
"""

from typing import Dict, Any, Optional
from core.database import get_connection
from api_v2.db_layer import administrative_subcase_db


# Privileged roles that can view any incident (regardless of subcase scope)
PRIVILEGED_VIEWER_ROLES = [
    "SOFTWARE_ADMIN",
    "COMPLAINT_SUPERVISOR",
    "ADMINISTRATION_ADMIN",
]


def can_view_incident(incident_id: int, current_user) -> bool:
    """
    Check if user is authorized to view incident details.
    
    Authorization rules:
    1. User is in PRIVILEGED_VIEWER_ROLES (can view any incident)
    2. User has an active subcase for this incident in their org unit scope
    
    Args:
        incident_id: Incident ID to check
        current_user: CurrentUser object with scopes and allowed_unit_ids
    
    Returns:
        True if authorized, False otherwise
    """
    # Check privileged roles first (roles is a list in CurrentUser)
    if hasattr(current_user, 'roles') and current_user.roles:
        for role in current_user.roles:
            if role in PRIVILEGED_VIEWER_ROLES:
                return True
    
    # Fallback: check via scopes if available
    if hasattr(current_user, 'scopes') and current_user.scopes:
        for scope in current_user.scopes:
            if scope.role_code in PRIVILEGED_VIEWER_ROLES:
                return True
    
    # Check if user has a subcase in their scope for this incident
    allowed_unit_ids = set(current_user.allowed_unit_ids) if current_user.allowed_unit_ids else set()
    
    return administrative_subcase_db.check_user_has_subcase_for_incident(
        incident_id=incident_id,
        allowed_unit_ids=allowed_unit_ids
    )


def get_incident_detail(incident_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch read-only incident details for workflow viewing.
    
    Returns fields needed by the workflow inbox "View" action:
    - incident_id, complaint_text
    - Classification fields (domain, category, subcategory, classification, severity)
    - Organizational fields (issuing_department_name, building, in_out)
    - Dates (feedback_received_date, created_at)
    - Patient info (patient_name)
    - Actions (immediate_action, taken_action)
    - Linked entities (target_departments, doctors, employees)
    
    Args:
        incident_id: Incident ID to fetch
    
    Returns:
        Dictionary with incident details, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Main incident query with all classification lookups
        query = """
            SELECT 
                c.IncidentRequestCaseID as incident_id,
                c.ComplaintText as complaint_text,
                c.ImmediateAction as immediate_action,
                c.TakenAction as taken_action,
                c.FeedbackRecievedDate as feedback_received_date,
                c.PatientName as patient_name,
                c.CreatedAt as created_at,
                c.isINPatient as is_inpatient,
                
                -- Issuing organizational unit
                org_unit.Name as issuing_department_name,
                
                -- Domain
                domain.DomainName as domain_name,
                
                -- Category
                category.CategoryName as category_name,
                
                -- SubCategory
                subcategory.SubCategoryName as subcategory_name,
                
                -- Classification
                classification.Classification_AR as classification_name,
                
                -- Severity
                severity.SeverityName as severity_name,
                
                -- Building
                building.BuildingName as building,
                
                -- IN/OUT status (from is_inpatient flag)
                CASE 
                    WHEN c.isINPatient = 1 THEN 'IN'
                    WHEN c.isINPatient = 0 THEN 'OUT'
                    ELSE NULL
                END as in_out
                
            FROM dbo.APP_IncidentCase c
            LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
            LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
            LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
            LEFT JOIN APP_LOOKUP_SUBCATEGORY subcategory ON c.SubCategoryID = subcategory.SubCategoryID
            LEFT JOIN APP_LOOKUP_CLASSIFICATION classification ON c.ClassificationID = classification.ClassificationID
            LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
            LEFT JOIN APP_LOOKUP_BUILDING building ON c.BuildingID = building.BuildingID
            WHERE c.IncidentRequestCaseID = ?
        """
        
        cursor.execute(query, (incident_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Build result dict
        columns = [column[0] for column in cursor.description]
        result = dict(zip(columns, row))
        
        # Format dates
        if result.get('feedback_received_date'):
            result['feedback_received_date'] = result['feedback_received_date'].strftime('%Y-%m-%d')
        if result.get('created_at'):
            result['created_at'] = result['created_at'].isoformat()
        
        # Fetch target departments (sections)
        target_dept_query = """
            SELECT 
                td.DepartmentID as id,
                sec_unit.Name as name
            FROM dbo.APP_IncidentCaseTargetDepartment td
            LEFT JOIN dbo.AdminsrationUnit sec_unit ON td.DepartmentID = sec_unit.UniqueID
            WHERE td.IncidentRequestCaseID = ?
            ORDER BY td.IsPrimary DESC, td.DepartmentID
        """
        cursor.execute(target_dept_query, (incident_id,))
        target_dept_rows = cursor.fetchall()
        
        result['target_departments'] = [
            {'id': row.id, 'name': row.name}
            for row in target_dept_rows
        ]
        
        # Fetch linked doctors
        doctor_query = """
            SELECT 
                d.DoctorID as doctor_id,
                d.DoctorName as name
            FROM dbo.APP_IncidentCaseDoctor d
            WHERE d.IncidentRequestCaseID = ?
            ORDER BY d.IsPrimary DESC, d.DoctorID
        """
        cursor.execute(doctor_query, (incident_id,))
        doctor_rows = cursor.fetchall()
        
        result['doctors'] = [
            {'doctor_id': row.doctor_id, 'name': row.name}
            for row in doctor_rows
        ]
        
        # Fetch linked employees
        employee_query = """
            SELECT 
                e.EmployeeID as employee_id,
                e.FullName as full_name
            FROM dbo.APP_IncidentCaseEmployee e
            WHERE e.IncidentRequestCaseID = ?
            ORDER BY e.IsPrimary DESC, e.EmployeeID
        """
        cursor.execute(employee_query, (incident_id,))
        employee_rows = cursor.fetchall()
        
        result['employees'] = [
            {'employee_id': row.employee_id, 'full_name': row.full_name}
            for row in employee_rows
        ]
        
        return result
        
    finally:
        cursor.close()
        conn.close()
