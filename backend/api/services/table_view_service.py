"""
Table View Service (Simplified for Actual Schema)
Handles data retrieval and filtering for the main TableView page.
Provides paginated, searchable, and filterable complaints/incidents list.

NOTE: This version works with the actual APP_IncidentCase schema which has:
- IncidentRequestCaseID (primary key)
- ComplaintText, ImmediateAction, TakenAction
- FeedbackRecievedDate, PatientName, DoctorName, DoctorID
- IssuingOrgUnitID (organizational unit)
- DomainID, CategoryID, SubCategoryID, ClassificationID
- SeverityID, StageID, HarmLevelID, CaseStatusID
- CreatedAt, CreatedByUserID, InOut, ClinicalRiskTypeID, FeedbackIntentTypeID, BuildingID
"""

from typing import Dict, List, Optional, Any, Literal
from datetime import datetime, date
import pyodbc
from dateutil.relativedelta import relativedelta
from io import BytesIO
from core.database import get_connection
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

from ..db_layer.database import get_connection


# ==================== HELPER FUNCTIONS ====================

def _get_workflow_status(incident_id: int, cursor=None) -> Optional[Dict[str, Any]]:
    """
    Get workflow status for an incident (subcases information).
    
    Returns:
        Dictionary with workflow status info or None if no subcases
        {
            "has_subcases": bool,
            "open_subcase_count": int,
            "force_closed": bool,
            "subcases": [
                {
                    "subcase_id": int,
                    "status": str,
                    "target_org_unit": str,
                    "target_org_unit_id": int
                },
                ...
            ]
        }
    """
    close_cursor = False
    if cursor is None:
        conn = get_connection()
        cursor = conn.cursor()
        close_cursor = True
    
    try:
        query = """
            SELECT 
                s.SubcaseID,
                s.Status,
                s.TargetOrgUnitID,
                org.Name as TargetOrgUnitName
            FROM dbo.APP_AdministrativeSubcase s
            LEFT JOIN AdminsrationUnit org ON s.TargetOrgUnitID = org.UniqueID
            WHERE s.IncidentRequestCaseID = ?
            ORDER BY s.CreatedAt ASC
        """
        
        cursor.execute(query, (incident_id,))
        rows = cursor.fetchall()
        
        if not rows:
            return None
        
        subcases = []
        open_count = 0
        force_closed_count = 0
        
        for row in rows:
            status = row.Status
            if status == 'FORCE_CLOSED':
                force_closed_count += 1
            elif status not in ['ADMIN_APPROVED', 'SECTION_DENIED']:
                open_count += 1
            
            subcases.append({
                "subcase_id": row.SubcaseID,
                "status": status,
                "target_org_unit": row.TargetOrgUnitName,
                "target_org_unit_id": row.TargetOrgUnitID
            })
        
        return {
            "has_subcases": True,
            "open_subcase_count": open_count,
            "force_closed": force_closed_count > 0,
            "subcases": subcases
        }
    
    finally:
        if close_cursor:
            cursor.close()
            conn.close()


# ==================== MAIN ENDPOINTS ====================

def get_complaints_paginated(
    search: Optional[str] = None,
    issuing_org_unit_id: Optional[int] = None,
    domain_id: Optional[int] = None,
    category_id: Optional[int] = None,
    severity_id: Optional[int] = None,
    stage_id: Optional[int] = None,
    harm_level_id: Optional[int] = None,
    case_status_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    sort_by: str = "FeedbackRecievedDate",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 50,
    view: str = "complete"
) -> Dict[str, Any]:
    """
    Fetch paginated complaints with search and filtering.
    
    Returns:
        Dictionary with complaints array, pagination info, and applied filters.
    """
    # Validate pagination
    if page < 1:
        raise ValueError("Page must be >= 1")
    if page_size < 1 or page_size > 500:
        raise ValueError("Page size must be between 1 and 500")
    
    # Validate sort parameters
    valid_sort_fields = [
        "FeedbackRecievedDate", "IncidentRequestCaseID", "SeverityID", 
        "CreatedAt", "PatientName", "id"
    ]
    if sort_by not in valid_sort_fields:
        sort_by = "FeedbackRecievedDate"
    
    if sort_order.lower() not in ["asc", "desc"]:
        sort_order = "desc"
    
    # Map 'id' to actual database column
    if sort_by == "id":
        sort_by = "IncidentRequestCaseID"
    
    # Build WHERE clause dynamically
    where_conditions = []
    params = []
    
    # Free-text search
    if search:
        search_condition = """(
            CAST(c.IncidentRequestCaseID AS VARCHAR) LIKE ? 
            OR c.PatientName LIKE ? 
            OR c.ComplaintText LIKE ?
        )"""
        where_conditions.append(search_condition)
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param])
    
    # Organizational unit filter
    if issuing_org_unit_id:
        where_conditions.append("c.IssuingOrgUnitID = ?")
        params.append(issuing_org_unit_id)
    
    # Domain filter
    if domain_id:
        where_conditions.append("c.DomainID = ?")
        params.append(domain_id)
    
    # Category filter
    if category_id:
        where_conditions.append("c.CategoryID = ?")
        params.append(category_id)
    
    # Severity filter
    if severity_id:
        where_conditions.append("c.SeverityID = ?")
        params.append(severity_id)
    
    # Stage filter
    if stage_id:
        where_conditions.append("c.StageID = ?")
        params.append(stage_id)
    
    # Harm level filter
    if harm_level_id:
        where_conditions.append("c.HarmLevelID = ?")
        params.append(harm_level_id)
    
    # Case status filter
    if case_status_id:
        where_conditions.append("c.CaseStatusID = ?")
        params.append(case_status_id)
    
    # Date filters
    if year:
        where_conditions.append("YEAR(c.FeedbackRecievedDate) = ?")
        params.append(year)
    
    if month:
        where_conditions.append("MONTH(c.FeedbackRecievedDate) = ?")
        params.append(month)
    
    if start_date:
        where_conditions.append("c.FeedbackRecievedDate >= ?")
        params.append(start_date)
    
    if end_date:
        where_conditions.append("c.FeedbackRecievedDate <= ?")
        params.append(end_date)
    
    # Combine WHERE conditions
    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)
    
    # Calculate offset for pagination
    offset = (page - 1) * page_size
    
    # Build ORDER BY clause with CAST for numeric sorting on ID field
    if sort_by == "IncidentRequestCaseID":
        order_by_clause = f"ORDER BY CAST(c.{sort_by} AS INT) {sort_order.upper()}"
    else:
        order_by_clause = f"ORDER BY c.{sort_by} {sort_order.upper()}"
    
    # SQL query for fetching records
    query = f"""
        SELECT 
            c.IncidentRequestCaseID as id,
            c.IncidentRequestCaseID as complaint_number,
            LEFT(c.ComplaintText, 150) as complaint_summary,
            c.ComplaintText as complaint_text,
            c.FeedbackRecievedDate as received_date,
            c.CreatedAt as created_at,
            c.PatientName as patient_name,
            
            -- Issuing organizational unit
            c.IssuingOrgUnitID as issuing_org_unit_id,
            org_unit.Name as issuing_org_unit_name,
            
            -- Domain
            c.DomainID as domain_id,
            domain.DomainName as domain_name,
            
            -- Category
            c.CategoryID as category_id,
            category.CategoryName as category_name,
            
            -- SubCategory
            c.SubCategoryID as subcategory_id,
            subcategory.SubCategoryName as subcategory_name,
            
            -- Classification
            c.ClassificationID as classification_id,
            classification.Classification_AR as classification_name,
            
            -- Severity
            c.SeverityID as severity_id,
            severity.SeverityName as severity_name,
            
            -- Stage
            c.StageID as stage_id,
            stage.StageName as stage_name,
            
            -- Harm level
            c.HarmLevelID as harm_level_id,
            harm.HarmLevel as harm_level,
            
            -- Case Status
            c.CaseStatusID as case_status_id,
            status.Name as status_name,
            
            -- Building
            c.BuildingID as building_id,
            building.BuildingName as building_name,
            
            -- Risk and Intent Types
            c.ClinicalRiskTypeID as clinical_risk_type_id,
            clinical_risk.Name as clinical_risk_type_name,
            c.FeedbackIntentTypeID as feedback_intent_type_id,
            feedback_intent.NameEn as feedback_intent_type_name,
            
            -- Source
            c.SourceID as source_id,
            source.SourceName as source_name,
            
            -- Explanation Status
            c.ExplanationStatusID as explanation_status_id,
            explanation_status.StatusName as explanation_status_name,
            
            -- Other fields
            c.ImmediateAction as immediate_action,
            c.TakenAction as taken_action,
            c.isINPatient as is_inpatient,
            c.CreatedByUserID as created_by_user_id
            
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
        LEFT JOIN APP_LOOKUP_SUBCATEGORY subcategory ON c.SubCategoryID = subcategory.SubCategoryID
        LEFT JOIN APP_LOOKUP_CLASSIFICATION classification ON c.ClassificationID = classification.ClassificationID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_BUILDING building ON c.BuildingID = building.BuildingID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE clinical_risk ON c.ClinicalRiskTypeID = clinical_risk.ClinicalRiskTypeID
        LEFT JOIN APP_LOOKUP_FEEDBACK_INTENT_TYPE feedback_intent ON c.FeedbackIntentTypeID = feedback_intent.FeedbackIntentTypeID
        LEFT JOIN APP_LOOKUP_SOURCE source ON c.SourceID = source.SourceID
        LEFT JOIN APP_LOOKUP_EXPLANATION_STATUS explanation_status ON c.ExplanationStatusID = explanation_status.StatusID
        {where_clause}
        {order_by_clause}
        OFFSET ? ROWS
        FETCH NEXT ? ROWS ONLY
    """
    
    # Add pagination params
    params.extend([offset, page_size])
    
    # Get total count
    count_query = f"""
        SELECT COUNT(*) as total
        FROM dbo.APP_IncidentCase c
        {where_clause}
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Execute count query
        cursor.execute(count_query, params[:-2])  # Exclude offset and page_size
        total_records = cursor.fetchone().total
        
        # Execute main query
        cursor.execute(query, params)
        columns = [column[0] for column in cursor.description]
        
        complaints = []
        for row in cursor.fetchall():
            complaint = dict(zip(columns, row))
            
            # Format dates
            if complaint.get('received_date'):
                complaint['received_date'] = complaint['received_date'].strftime('%Y-%m-%d')
            if complaint.get('created_at'):
                complaint['created_at'] = complaint['created_at'].isoformat()
            
            # Calculate days_open if not completed
            if complaint.get('received_date'):
                try:
                    received = datetime.strptime(complaint['received_date'], '%Y-%m-%d')
                    complaint['days_open'] = (datetime.now() - received).days
                except:
                    complaint['days_open'] = None
            else:
                complaint['days_open'] = None
            
            # Add workflow_status information
            complaint['workflow_status'] = _get_workflow_status(complaint['id'], cursor)
            
            complaints.append(complaint)
        
        total_pages = (total_records + page_size - 1) // page_size
        
        # Build filters_applied dict
        filters_applied = {
            'search': search,
            'issuing_org_unit_id': issuing_org_unit_id,
            'domain_id': domain_id,
            'category_id': category_id,
            'severity_id': severity_id,
            'stage_id': stage_id,
            'harm_level_id': harm_level_id,
            'case_status_id': case_status_id,
            'year': year,
            'month': month,
            'start_date': start_date,
            'end_date': end_date
        }
        
        return {
            'complaints': complaints,
            'pagination': {
                'page': page,
                'page_size': page_size,
                'total_records': total_records,
                'total_pages': total_pages
            },
            'filters_applied': filters_applied,
            'view': view
        }
        
    finally:
        cursor.close()
        conn.close()


def get_filter_options(include_counts: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch available filter options for dropdown population.
    
    Args:
        include_counts: If True, include record count for each option.
        
    Returns:
        Dictionary with arrays for organizational units, domains, categories, severities, etc.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        result = {}
        
        # Issuing Organizational Units
        org_query = """
            SELECT DISTINCT
                u.UniqueID as id,
                u.Name as name,
                u.ParentID as parent_id
                """ + (", COUNT(c.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM AdminsrationUnit u
            """ + ("LEFT JOIN APP_IncidentCase c ON u.UniqueID = c.IssuingOrgUnitID" if include_counts else "") + """
            WHERE u.Frozen = 0
            """ + ("GROUP BY u.UniqueID, u.Name, u.ParentID" if include_counts else "") + """
            ORDER BY u.Name
        """
        cursor.execute(org_query)
        result['issuing_org_units'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Domains
        domain_query = """
            SELECT 
                d.DomainID as id,
                d.DomainName as name
                """ + (", COUNT(c.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_DOMAIN d
            """ + ("LEFT JOIN APP_IncidentCase c ON d.DomainID = c.DomainID" if include_counts else "") + """
            """ + ("GROUP BY d.DomainID, d.DomainName" if include_counts else "") + """
            ORDER BY d.DomainName
        """
        cursor.execute(domain_query)
        result['domains'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Categories
        category_query = """
            SELECT 
                c.CategoryID as id,
                c.CategoryName as name,
                c.DomainID as domain_id
                """ + (", COUNT(ic.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_CATEGORY c
            """ + ("LEFT JOIN APP_IncidentCase ic ON c.CategoryID = ic.CategoryID" if include_counts else "") + """
            """ + ("GROUP BY c.CategoryID, c.CategoryName, c.DomainID" if include_counts else "") + """
            ORDER BY c.CategoryName
        """
        cursor.execute(category_query)
        result['categories'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Severities
        severity_query = """
            SELECT 
                s.SeverityID as id,
                s.SeverityName as name
                """ + (", COUNT(c.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_SEVERITY s
            """ + ("LEFT JOIN APP_IncidentCase c ON s.SeverityID = c.SeverityID" if include_counts else "") + """
            WHERE s.IsActive = 1
            """ + ("GROUP BY s.SeverityID, s.SeverityName" if include_counts else "") + """
            ORDER BY s.SeverityOrder
        """
        cursor.execute(severity_query)
        result['severities'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Stages
        stage_query = """
            SELECT 
                s.StageID as id,
                s.StageName as name
                """ + (", COUNT(c.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_CASE_STAGE s
            """ + ("LEFT JOIN APP_IncidentCase c ON s.StageID = c.StageID" if include_counts else "") + """
            """ + ("GROUP BY s.StageID, s.StageName" if include_counts else "") + """
            ORDER BY s.StageID
        """
        cursor.execute(stage_query)
        result['stages'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Harm Levels
        harm_query = """
            SELECT 
                h.HarmID as id,
                h.HarmLevel as name
                """ + (", COUNT(c.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_HARM_LEVEL h
            """ + ("LEFT JOIN APP_IncidentCase c ON h.HarmID = c.HarmLevelID" if include_counts else "") + """
            """ + ("GROUP BY h.HarmID, h.HarmLevel" if include_counts else "") + """
            ORDER BY h.HarmID
        """
        cursor.execute(harm_query)
        result['harm_levels'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Statuses
        status_query = """
            SELECT 
                s.CaseStatusID as id,
                s.Name as name
                """ + (", COUNT(c.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_CASE_STATUS s
            """ + ("LEFT JOIN APP_IncidentCase c ON s.CaseStatusID = c.CaseStatusID" if include_counts else "") + """
            """ + ("GROUP BY s.CaseStatusID, s.Name" if include_counts else "") + """
            WHERE s.IsActive = 1
            ORDER BY s.DisplayOrder
        """
        cursor.execute(status_query)
        result['statuses'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Classifications EN
        classification_en_query = """
            SELECT 
                c.ClassificationID as id,
                c.Classification_AR as name
                """ + (", COUNT(ic.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_CLASSIFICATION c
            """ + ("LEFT JOIN APP_IncidentCase ic ON c.ClassificationID = ic.ClassificationID" if include_counts else "") + """
            """ + ("GROUP BY c.ClassificationID, c.Classification_AR" if include_counts else "") + """
            ORDER BY c.Classification_AR
        """
        cursor.execute(classification_en_query)
        result['classifications_en'] = [dict(zip([col[0] for col in cursor.description], row)) for row in cursor.fetchall()]
        
        # Years
        years_query = """
            SELECT DISTINCT YEAR(FeedbackRecievedDate) as year
            FROM APP_IncidentCase
            WHERE FeedbackRecievedDate IS NOT NULL
            ORDER BY year DESC
        """
        cursor.execute(years_query)
        result['years'] = [row[0] for row in cursor.fetchall()]
        
        return result
        
    finally:
        cursor.close()
        conn.close()


def get_complaint_by_id(complaint_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch full details of a single complaint record.
    
    Args:
        complaint_id: The IncidentRequestCaseID to retrieve.
        
    Returns:
        Dictionary with full complaint details, or None if not found.
    """
    query = """
        SELECT 
            c.IncidentRequestCaseID as id,
            c.ComplaintText as complaint_text,
            c.ImmediateAction as immediate_action,
            c.TakenAction as taken_action,
            c.FeedbackRecievedDate as received_date,
            c.PatientName as patient_name,
            c.CreatedAt as created_at,
            c.CreatedByUserID as created_by_user_id,
            c.isINPatient as is_inpatient,
            
            -- Issuing organizational unit
            c.IssuingOrgUnitID as issuing_org_unit_id,
            org_unit.Name as issuing_org_unit_name,
            
            -- Domain
            c.DomainID as domain_id,
            domain.DomainName as domain_name,
            
            -- Category
            c.CategoryID as category_id,
            category.CategoryName as category_name,
            
            -- SubCategory
            c.SubCategoryID as subcategory_id,
            subcategory.SubCategoryName as subcategory_name,
            
            -- Classification
            c.ClassificationID as classification_id,
            classification.Classification_AR as classification_name,
            
            -- Severity
            c.SeverityID as severity_id,
            severity.SeverityName as severity_name,
            
            -- Stage
            c.StageID as stage_id,
            stage.StageName as stage_name,
            
            -- Harm level
            c.HarmLevelID as harm_level_id,
            harm.HarmLevel as harm_level,
            
            -- Case Status
            c.CaseStatusID as case_status_id,
            status.Name as status_name,
            
            -- Building
            c.BuildingID as building_id,
            building.BuildingName as building_name,
            
            -- Risk and Intent Types
            c.ClinicalRiskTypeID as clinical_risk_type_id,
            clinical_risk.Name as clinical_risk_type_name,
            c.FeedbackIntentTypeID as feedback_intent_type_id,
            feedback_intent.NameEn as feedback_intent_type_name,
            
            -- Source
            c.SourceID as source_id,
            source.SourceName as source_name,
            
            -- Explanation Status
            c.ExplanationStatusID as explanation_status_id,
            explanation_status.StatusName as explanation_status_name
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
        LEFT JOIN APP_LOOKUP_SUBCATEGORY subcategory ON c.SubCategoryID = subcategory.SubCategoryID
        LEFT JOIN APP_LOOKUP_CLASSIFICATION classification ON c.ClassificationID = classification.ClassificationID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_BUILDING building ON c.BuildingID = building.BuildingID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE clinical_risk ON c.ClinicalRiskTypeID = clinical_risk.ClinicalRiskTypeID
        LEFT JOIN APP_LOOKUP_FEEDBACK_INTENT_TYPE feedback_intent ON c.FeedbackIntentTypeID = feedback_intent.FeedbackIntentTypeID
        LEFT JOIN APP_LOOKUP_SOURCE source ON c.SourceID = source.SourceID
        LEFT JOIN APP_LOOKUP_EXPLANATION_STATUS explanation_status ON c.ExplanationStatusID = explanation_status.StatusID
        WHERE c.IncidentRequestCaseID = ?
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, (complaint_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = [column[0] for column in cursor.description]
        complaint = dict(zip(columns, row))
        
        # Format dates
        if complaint.get('received_date'):
            complaint['received_date'] = complaint['received_date'].strftime('%Y-%m-%d')
        if complaint.get('created_at'):
            complaint['created_at'] = complaint['created_at'].isoformat()
        
        # Fetch target departments
        target_dept_query = """
            SELECT 
                td.DepartmentID as section_id,
                sec_unit.Name as section_name,
                dept_unit.UniqueID as department_id,
                dept_unit.Name as department_name,
                admin_unit.UniqueID as administration_id,
                admin_unit.Name as administration_name,
                td.IsPrimary as is_primary
            FROM dbo.APP_IncidentCaseTargetDepartment td
            LEFT JOIN dbo.AdminsrationUnit sec_unit ON td.DepartmentID = sec_unit.UniqueID      -- Section (leaf)
            LEFT JOIN dbo.AdminsrationUnit dept_unit ON sec_unit.ParentID = dept_unit.UniqueID   -- Department (parent)
            LEFT JOIN dbo.AdminsrationUnit admin_unit ON dept_unit.ParentID = admin_unit.UniqueID -- Administration (grandparent)
            WHERE td.IncidentRequestCaseID = ?
            ORDER BY td.IsPrimary DESC, td.DepartmentID
        """
        cursor.execute(target_dept_query, (complaint_id,))
        target_dept_rows = cursor.fetchall()
        
        target_departments = []
        for dept_row in target_dept_rows:
            target_departments.append({
                'section_id': dept_row.section_id,
                'section_name': dept_row.section_name,
                'department_id': dept_row.department_id,
                'department_name': dept_row.department_name,
                'administration_id': dept_row.administration_id,
                'administration_name': dept_row.administration_name,
                'is_primary': bool(dept_row.is_primary)
            })
        
        complaint['target_departments'] = target_departments
        
        # Fetch linked employees (supervisors/workers)
        employee_query = """
            SELECT 
                e.EmployeeID as employee_id,
                e.FullName as full_name,
                e.JobTitle as job_title,
                e.IsPrimary as is_primary
            FROM dbo.APP_IncidentCaseEmployee e
            WHERE e.IncidentRequestCaseID = ?
            ORDER BY e.IsPrimary DESC, e.EmployeeID
        """
        cursor.execute(employee_query, (complaint_id,))
        employee_rows = cursor.fetchall()
        
        employees = []
        for emp_row in employee_rows:
            employees.append({
                'employee_id': emp_row.employee_id,
                'full_name': emp_row.full_name,
                'job_title': emp_row.job_title if hasattr(emp_row, 'job_title') else None,
                'is_primary': bool(emp_row.is_primary) if emp_row.is_primary is not None else False
            })
        
        complaint['employees'] = employees
        
        # Fetch linked doctors
        doctor_query = """
            SELECT 
                d.DoctorID as doctor_id,
                d.DoctorName as doctor_name,
                d.IsPrimary as is_primary
            FROM dbo.APP_IncidentCaseDoctor d
            WHERE d.IncidentRequestCaseID = ?
            ORDER BY d.IsPrimary DESC, d.DoctorID
        """
        cursor.execute(doctor_query, (complaint_id,))
        doctor_rows = cursor.fetchall()
        
        doctors = []
        for doc_row in doctor_rows:
            doctors.append({
                'doctor_id': doc_row.doctor_id,
                'doctor_name': doc_row.doctor_name,
                'is_primary': bool(doc_row.is_primary) if doc_row.is_primary is not None else False
            })
        
        complaint['doctors'] = doctors
        
        return complaint
        
    finally:
        cursor.close()
        conn.close()


def get_complaints_count(
    search: Optional[str] = None,
    issuing_org_unit_id: Optional[int] = None,
    domain_id: Optional[int] = None,
    category_id: Optional[int] = None,
    severity_id: Optional[int] = None,
    stage_id: Optional[int] = None,
    harm_level_id: Optional[int] = None,
    case_status_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get count of complaints matching filters (for export preview).
    
    Returns:
        Dictionary with total_count and filters_applied.
    """
    # Build WHERE clause (same as get_complaints_paginated)
    where_conditions = []
    params = []
    
    if search:
        search_condition = """(
            CAST(c.IncidentRequestCaseID AS VARCHAR) LIKE ? 
            OR c.PatientName LIKE ? 
            OR c.ComplaintText LIKE ?
        )"""
        where_conditions.append(search_condition)
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param])
    
    if issuing_org_unit_id:
        where_conditions.append("c.IssuingOrgUnitID = ?")
        params.append(issuing_org_unit_id)
    
    if domain_id:
        where_conditions.append("c.DomainID = ?")
        params.append(domain_id)
    
    if category_id:
        where_conditions.append("c.CategoryID = ?")
        params.append(category_id)
    
    if severity_id:
        where_conditions.append("c.SeverityID = ?")
        params.append(severity_id)
    
    if stage_id:
        where_conditions.append("c.StageID = ?")
        params.append(stage_id)
    
    if harm_level_id:
        where_conditions.append("c.HarmLevelID = ?")
        params.append(harm_level_id)
    
    if case_status_id:
        where_conditions.append("c.CaseStatusID = ?")
        params.append(case_status_id)
    
    if year:
        where_conditions.append("YEAR(c.FeedbackRecievedDate) = ?")
        params.append(year)
    
    if month:
        where_conditions.append("MONTH(c.FeedbackRecievedDate) = ?")
        params.append(month)
    
    if start_date:
        where_conditions.append("c.FeedbackRecievedDate >= ?")
        params.append(start_date)
    
    if end_date:
        where_conditions.append("c.FeedbackRecievedDate <= ?")
        params.append(end_date)
    
    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)
    
    query = f"""
        SELECT COUNT(*) as total
        FROM dbo.APP_IncidentCase c
        {where_clause}
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, params)
        total_count = cursor.fetchone().total
        
        filters_applied = {
            'search': search,
            'issuing_org_unit_id': issuing_org_unit_id,
            'domain_id': domain_id,
            'category_id': category_id,
            'severity_id': severity_id,
            'stage_id': stage_id,
            'harm_level_id': harm_level_id,
            'case_status_id': case_status_id,
            'year': year,
            'month': month,
            'start_date': start_date,
            'end_date': end_date
        }
        
        return {
            'total_count': total_count,
            'filters_applied': filters_applied
        }
        
    finally:
        cursor.close()
        conn.close()


def export_complaints_excel(
    search: Optional[str] = None,
    issuing_org_unit_id: Optional[int] = None,
    domain_id: Optional[int] = None,
    category_id: Optional[int] = None,
    severity_id: Optional[int] = None,
    stage_id: Optional[int] = None,
    harm_level_id: Optional[int] = None,
    case_status_id: Optional[int] = None,
    year: Optional[int] = None,
    month: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> BytesIO:
    """
    Export filtered complaints as Excel file.
    
    Returns:
        BytesIO object containing the Excel file.
    """
    # Build WHERE clause (same logic as get_complaints_paginated)
    where_conditions = []
    params = []
    
    if search:
        search_condition = """(
            CAST(c.IncidentRequestCaseID AS VARCHAR) LIKE ? 
            OR c.PatientName LIKE ? 
            OR c.ComplaintText LIKE ?
        )"""
        where_conditions.append(search_condition)
        search_param = f"%{search}%"
        params.extend([search_param, search_param, search_param])
    
    if issuing_org_unit_id:
        where_conditions.append("c.IssuingOrgUnitID = ?")
        params.append(issuing_org_unit_id)
    
    if domain_id:
        where_conditions.append("c.DomainID = ?")
        params.append(domain_id)
    
    if category_id:
        where_conditions.append("c.CategoryID = ?")
        params.append(category_id)
    
    if severity_id:
        where_conditions.append("c.SeverityID = ?")
        params.append(severity_id)
    
    if stage_id:
        where_conditions.append("c.StageID = ?")
        params.append(stage_id)
    
    if harm_level_id:
        where_conditions.append("c.HarmLevelID = ?")
        params.append(harm_level_id)
    
    if case_status_id:
        where_conditions.append("c.CaseStatusID = ?")
        params.append(case_status_id)
    
    if year:
        where_conditions.append("YEAR(c.FeedbackRecievedDate) = ?")
        params.append(year)
    
    if month:
        where_conditions.append("MONTH(c.FeedbackRecievedDate) = ?")
        params.append(month)
    
    if start_date:
        where_conditions.append("c.FeedbackRecievedDate >= ?")
        params.append(start_date)
    
    if end_date:
        where_conditions.append("c.FeedbackRecievedDate <= ?")
        params.append(end_date)
    
    where_clause = ""
    if where_conditions:
        where_clause = "WHERE " + " AND ".join(where_conditions)
    
    # Query to fetch all matching records with all required fields
    # Note: Some lookup tables may not exist yet, using COALESCE for optional fields
    query = f"""
        SELECT 
            c.FeedbackRecievedDate as received_date,
            c.IncidentRequestCaseID as complaint_number,
            c.PatientName as patient_name,
            issuing_org.Name as issuing_org_unit_name,
            concerned_org.Name as concerned_org_unit_name,
            source.SourceNameAr as source_name,
            CASE 
                WHEN c.FeedbackIntentTypeID = 1 THEN N'شكوى'
                WHEN c.FeedbackIntentTypeID = 2 THEN N'ملاحظة'
                WHEN c.FeedbackIntentTypeID = 3 THEN N'اقتراح'
                WHEN c.FeedbackIntentTypeID = 4 THEN N'استفسار'
                ELSE N'غير محدد'
            END as feedback_type,
            domain.DomainName as domain_name,
            category.CategoryName as category_name,
            subcat.SubCategoryName as subcategory_name,
            classif.Classification_AR as classification_name,
            c.ComplaintText as complaint_text,
            c.ImmediateAction as immediate_action,
            c.TakenAction as taken_action,
            severity.SeverityName as severity_name,
            stage.StageName as stage_name,
            harm.HarmLevel as harm_level,
            status.Name as status_name,
            risk_type.Name as feedback_risk_type
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit issuing_org ON c.IssuingOrgUnitID = issuing_org.UniqueID
        LEFT JOIN AdminsrationUnit concerned_org ON c.BuildingID = concerned_org.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
        LEFT JOIN APP_LOOKUP_SUBCATEGORY subcat ON c.SubCategoryID = subcat.SubCategoryID
        LEFT JOIN APP_LOOKUP_CLASSIFICATION classif ON c.ClassificationID = classif.ClassificationID
        LEFT JOIN APP_LOOKUP_SOURCE source ON c.SourceID = source.SourceID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
        LEFT JOIN APP_LOOKUP_CLINICAL_RISK_TYPE risk_type ON c.ClinicalRiskTypeID = risk_type.ClinicalRiskTypeID
        {where_clause}
        ORDER BY c.FeedbackRecievedDate DESC
    """
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(query, params)
        rows = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        
        # Create Excel workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Complaints Export"
        
        # Set worksheet to Right-to-Left for Arabic text
        ws.sheet_view.rightToLeft = True
        
        # Define header style
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF", size=12)
        header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        # Write headers (Arabic translations) - Exact order as specified
        headers_arabic = {
            'received_date': 'تاريخ تلقي الملاحظة',
            'complaint_number': 'الرقم',
            'patient_name': 'اسم المريض',
            'issuing_org_unit_name': 'قسم الصادر',
            'concerned_org_unit_name': 'قسم المعني',
            'source_name': 'المصدر',
            'feedback_type': 'النوع (Feedback Type)',
            'domain_name': 'Domain',
            'category_name': 'Category',
            'subcategory_name': 'SubCategory',
            'classification_name': 'New Classification',
            'complaint_text': 'محتوى الشكوى (Raw Content)',
            'immediate_action': 'Immediate Action',
            'taken_action': 'الإجراءات المتخذة',
            'severity_name': 'Severity',
            'stage_name': 'Stage',
            'harm_level': 'Harm',
            'status_name': 'Status',
            'feedback_risk_type': 'FeedbackRiskType'
        }
        
        for col_idx, col_name in enumerate(columns, start=1):
            cell = ws.cell(row=1, column=col_idx)
            cell.value = headers_arabic.get(col_name, col_name)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = header_alignment
        
        # Write data rows
        for row_idx, row in enumerate(rows, start=2):
            for col_idx, value in enumerate(row, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                
                # Format dates
                if isinstance(value, (datetime, date)):
                    cell.value = value.strftime('%Y-%m-%d') if isinstance(value, date) else value.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    cell.value = value
                
                # Center align both horizontally and vertically for better readability
                cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        # Auto-adjust column widths
        for col_idx, col_name in enumerate(columns, start=1):
            column_letter = ws.cell(row=1, column=col_idx).column_letter
            if col_name in ['complaint_text', 'immediate_action', 'taken_action']:
                ws.column_dimensions[column_letter].width = 50
            elif col_name in ['patient_name', 'issuing_org_unit_name', 'concerned_org_unit_name', 'classification_name']:
                ws.column_dimensions[column_letter].width = 25
            elif col_name in ['domain_name', 'category_name', 'subcategory_name']:
                ws.column_dimensions[column_letter].width = 20
            else:
                ws.column_dimensions[column_letter].width = 18
        
        # Set row height for header
        ws.row_dimensions[1].height = 30
        
        # Save to BytesIO
        excel_file = BytesIO()
        wb.save(excel_file)
        excel_file.seek(0)
        
        return excel_file
        
    finally:
        cursor.close()
        conn.close()


def export_complaints(
    export_format: Literal['csv', 'json'],
    filters: Dict[str, Any],
    columns: List[str],
    include_patient_identifiers: bool = False,
    language: str = 'en'
) -> Dict[str, Any]:
    """
    Export filtered complaints as CSV or JSON.
    
    NOTE: This function prepares export metadata. Actual file generation
    should be handled by a background job or streaming response.
    """
    if export_format not in ['csv', 'json']:
        raise ValueError("Format must be 'csv' or 'json'")
    
    # Get count of records to export
    count_result = get_complaints_count(**filters)
    record_count = count_result['total_count']
    
    # Generate export ID (timestamp-based)
    from datetime import datetime
    export_id = f"exp-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Determine file name
    start_date = filters.get('start_date', '')
    end_date = filters.get('end_date', '')
    if start_date and end_date:
        file_name = f"Complaints_Export_{start_date}_to_{end_date}.{export_format}"
    else:
        file_name = f"Complaints_Export_{datetime.now().strftime('%Y%m%d')}.{export_format}"
    
    return {
        'export_id': export_id,
        'file_name': file_name,
        'file_size_bytes': record_count * 500,  # Rough estimate
        'download_url': f"/api/complaints/download/{export_id}",
        'record_count': record_count,
        'generated_at': datetime.now().isoformat(),
        'expires_at': (datetime.now() + relativedelta(days=1)).isoformat(),
        'audit_logged': True,
        'status': 'pending'
    }


def get_table_views() -> Dict[str, Any]:
    """
    Get predefined table view configurations.
    
    Returns:
        Dictionary with views array and default_view.
    """
    views = [
        {
            'view_id': 'complete',
            'view_name': 'Complete View',
            'view_name_ar': 'عرض كامل',
            'columns': [
                'complaint_number',
                'received_date',
                'patient_name',
                'issuing_org_unit_name',
                'domain_name',
                'category_name',
                'severity_name',
                'harm_level',
                'status_name',
                'days_open'
            ],
            'default_sort': 'received_date',
            'default_sort_order': 'desc'
        },
        {
            'view_id': 'simplified',
            'view_name': 'Simplified View',
            'view_name_ar': 'عرض مبسط',
            'columns': [
                'complaint_number',
                'received_date',
                'issuing_org_unit_name',
                'domain_name',
                'severity_name',
                'status_name'
            ],
            'default_sort': 'received_date',
            'default_sort_order': 'desc'
        }
    ]
    
    return {
        'views': views,
        'default_view': 'complete'
    }


def bulk_import_records_from_excel(file_content: bytes) -> Dict[str, Any]:
    """
    Import historical records from Excel file.
    
    SPECIAL RULES:
    - All imported records are inserted as: CaseStatusID=3 (Closed), ExplanationStatusID=4 (No Explanation Needed)
    - Bypasses FSM workflow
    - Uses same Excel template as export_complaints_excel()
    
    Args:
        file_content: Raw bytes of the uploaded Excel file
        
    Returns:
        Dictionary with import statistics: total_rows, inserted_count, failed_count, errors
    """
    from openpyxl import load_workbook
    from io import BytesIO
    
    # Statistics tracking
    inserted_count = 0
    failed_count = 0
    errors = []
    
    try:
        # Load Excel workbook
        wb = load_workbook(BytesIO(file_content), data_only=True)
        ws = wb.active
        
        # Expected column headers (Arabic from export template)
        expected_headers = {
            'تاريخ تلقي الملاحظة': 'received_date',
            'الرقم': 'complaint_number',
            'اسم المريض': 'patient_name',
            'قسم الصادر': 'issuing_org_unit_name',
            'قسم المعني': 'concerned_org_unit_name',
            'المصدر': 'source_name',
            'النوع (Feedback Type)': 'feedback_type',
            'Domain': 'domain_name',
            'Category': 'category_name',
            'SubCategory': 'subcategory_name',
            'New Classification': 'classification_name',
            'محتوى الشكوى (Raw Content)': 'complaint_text',
            'Immediate Action': 'immediate_action',
            'الإجراءات المتخذة': 'taken_action',
            'Severity': 'severity_name',
            'Stage': 'stage_name',
            'Harm': 'harm_level',
            'Status': 'status_name',
            'FeedbackRiskType': 'feedback_risk_type'
        }
        
        # Read header row (row 1)
        header_row = ws[1]
        header_map = {}
        for col_idx, cell in enumerate(header_row, start=1):
            if cell.value and cell.value in expected_headers:
                header_map[col_idx] = expected_headers[cell.value]
        
        # Validate headers
        if len(header_map) < 10:  # At least 10 essential columns
            return {
                'success': False,
                'error': 'INVALID_TEMPLATE',
                'message': 'Excel template does not match expected format. Please use the export template.',
                'inserted_count': 0,
                'failed_count': 0,
                'errors': []
            }
        
        # Get database connection for lookups
        conn = get_connection()
        cursor = conn.cursor()
        
        # Build lookup caches to avoid repeated DB queries
        lookup_caches = _build_lookup_caches(cursor)
        
        # Process each data row (starting from row 2)
        total_rows = ws.max_row - 1  # Exclude header
        
        for row_idx in range(2, ws.max_row + 1):
            row = ws[row_idx]
            
            try:
                # Extract row data
                row_data = {}
                for col_idx, field_name in header_map.items():
                    cell_value = row[col_idx - 1].value
                    row_data[field_name] = cell_value
                
                # Map Excel row to database payload
                payload = _map_excel_row_to_payload(row_data, lookup_caches, cursor)
                
                if payload.get('error'):
                    failed_count += 1
                    errors.append({
                        'row': row_idx,
                        'error': payload['error'],
                        'complaint_text': row_data.get('complaint_text', '')[:50]
                    })
                    continue
                
                # FORCE CLOSED STATE (bypass FSM)
                payload['case_status_id'] = 3  # Closed
                payload['explanation_status_id'] = 4  # No Explanation Needed
                
                # Insert record directly into database (bypass FSM)
                result = _insert_historical_record(payload, cursor, conn)
                
                if result.get('success'):
                    inserted_count += 1
                else:
                    failed_count += 1
                    errors.append({
                        'row': row_idx,
                        'error': result.get('error', 'Unknown error'),
                        'complaint_text': row_data.get('complaint_text', '')[:50]
                    })
            
            except Exception as row_error:
                failed_count += 1
                errors.append({
                    'row': row_idx,
                    'error': str(row_error),
                    'complaint_text': row_data.get('complaint_text', '')[:50] if 'row_data' in locals() else ''
                })
        
        cursor.close()
        conn.close()
        
        return {
            'success': True,
            'total_rows': total_rows,
            'inserted_count': inserted_count,
            'failed_count': failed_count,
            'errors': errors[:50]  # Limit to first 50 errors
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': 'FILE_PROCESSING_ERROR',
            'message': f'Failed to process Excel file: {str(e)}',
            'inserted_count': 0,
            'failed_count': 0,
            'errors': []
        }


def _build_lookup_caches(cursor) -> Dict[str, Dict]:
    """Build lookup caches for faster row processing."""
    caches = {}
    
    # Cache organizational units
    cursor.execute("SELECT UniqueID, Name FROM AdminsrationUnit WHERE Frozen = 0")
    caches['org_units'] = {row.Name.strip().upper(): row.UniqueID for row in cursor.fetchall()}
    
    # Cache domains
    cursor.execute("SELECT DomainID, DomainName FROM APP_LOOKUP_DOMAIN")
    caches['domains'] = {row.DomainName.strip().upper(): row.DomainID for row in cursor.fetchall()}
    
    # Cache categories
    cursor.execute("SELECT CategoryID, CategoryName FROM APP_LOOKUP_CATEGORY")
    caches['categories'] = {row.CategoryName.strip().upper(): row.CategoryID for row in cursor.fetchall()}
    
    # Cache severities
    cursor.execute("SELECT SeverityID, SeverityName FROM APP_LOOKUP_SEVERITY WHERE IsActive = 1")
    caches['severities'] = {row.SeverityName.strip().upper(): row.SeverityID for row in cursor.fetchall()}
    
    # Cache stages
    cursor.execute("SELECT StageID, StageName FROM APP_LOOKUP_CASE_STAGE")
    caches['stages'] = {row.StageName.strip().upper(): row.StageID for row in cursor.fetchall()}
    
    # Cache harm levels
    cursor.execute("SELECT HarmID, HarmLevel FROM APP_LOOKUP_HARM_LEVEL")
    caches['harm_levels'] = {row.HarmLevel.strip().upper(): row.HarmID for row in cursor.fetchall()}
    
    return caches


def _map_excel_row_to_payload(row_data: Dict, caches: Dict, cursor) -> Dict[str, Any]:
    """Map Excel row data to database insert payload."""
    try:
        payload = {}
        
        # Required text fields
        payload['complaint_text'] = row_data.get('complaint_text', '').strip() if row_data.get('complaint_text') else None
        if not payload['complaint_text']:
            return {'error': 'Missing complaint text'}
        
        payload['immediate_action'] = row_data.get('immediate_action', '').strip() if row_data.get('immediate_action') else None
        payload['taken_action'] = row_data.get('taken_action', '').strip() if row_data.get('taken_action') else None
        payload['patient_name'] = row_data.get('patient_name', '').strip() if row_data.get('patient_name') else None
        
        # Date
        received_date = row_data.get('received_date')
        if received_date:
            if isinstance(received_date, datetime):
                payload['feedback_received_date'] = received_date.strftime('%Y-%m-%d')
            elif isinstance(received_date, str):
                payload['feedback_received_date'] = received_date
            else:
                payload['feedback_received_date'] = str(received_date)
        else:
            payload['feedback_received_date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Lookup organizational unit
        org_name = row_data.get('issuing_org_unit_name', '').strip().upper()
        if org_name and org_name in caches['org_units']:
            payload['issuing_department_id'] = caches['org_units'][org_name]
        else:
            payload['issuing_department_id'] = 1  # Default
        
        # Lookup domain
        domain_name = row_data.get('domain_name', '').strip().upper()
        if domain_name and domain_name in caches['domains']:
            payload['domain_id'] = caches['domains'][domain_name]
        else:
            return {'error': f'Domain not found: {row_data.get("domain_name")}'}
        
        # Lookup category
        category_name = row_data.get('category_name', '').strip().upper()
        if category_name and category_name in caches['categories']:
            payload['category_id'] = caches['categories'][category_name]
        else:
            return {'error': f'Category not found: {row_data.get("category_name")}'}
        
        # Subcategory and classification (use numeric IDs from Excel)
        payload['subcategory_id'] = int(row_data.get('subcategory_name', 1)) if row_data.get('subcategory_name') else 1
        payload['classification_id'] = int(row_data.get('classification_name', 1)) if row_data.get('classification_name') else 1
        
        # Lookup severity
        severity_name = row_data.get('severity_name', '').strip().upper()
        if severity_name and severity_name in caches['severities']:
            payload['severity_id'] = caches['severities'][severity_name]
        else:
            payload['severity_id'] = 1  # Default
        
        # Lookup stage
        stage_name = row_data.get('stage_name', '').strip().upper()
        if stage_name and stage_name in caches['stages']:
            payload['stage_id'] = caches['stages'][stage_name]
        else:
            payload['stage_id'] = 1  # Default
        
        # Lookup harm level
        harm_name = row_data.get('harm_level', '').strip().upper()
        if harm_name and harm_name in caches['harm_levels']:
            payload['harm_id'] = caches['harm_levels'][harm_name]
        else:
            payload['harm_id'] = 1  # Default
        
        # Clinical risk type (from feedback_risk_type column)
        risk_type = row_data.get('feedback_risk_type', '').strip()
        if 'Red' in risk_type or 'راية' in risk_type:
            payload['clinical_risk_type_id'] = 2  # Red Flag
        elif 'Never' in risk_type or 'حدث' in risk_type:
            payload['clinical_risk_type_id'] = 3  # Never Event
        else:
            payload['clinical_risk_type_id'] = 1  # General
        
        # Feedback intent type (from feedback_type column)
        feedback_type = row_data.get('feedback_type', '').strip()
        if 'شكوى' in feedback_type or 'Complaint' in feedback_type:
            payload['feedback_intent_type_id'] = 1
        elif 'ملاحظة' in feedback_type or 'Observation' in feedback_type:
            payload['feedback_intent_type_id'] = 2
        elif 'اقتراح' in feedback_type or 'Suggestion' in feedback_type:
            payload['feedback_intent_type_id'] = 3
        elif 'استفسار' in feedback_type or 'Inquiry' in feedback_type:
            payload['feedback_intent_type_id'] = 4
        else:
            payload['feedback_intent_type_id'] = 1  # Default to complaint
        
        # Source (default)
        payload['source_id'] = 1
        
        return payload
    
    except Exception as e:
        return {'error': f'Mapping error: {str(e)}'}


def _insert_historical_record(payload: Dict[str, Any], cursor, conn) -> Dict[str, Any]:
    """
    Insert a historical record directly into database (bypass FSM).
    
    CRITICAL: All historical records are inserted as CLOSED with NO EXPLANATION NEEDED.
    """
    try:
        # Insert into APP_IncidentCase
        insert_query = """
            INSERT INTO dbo.APP_IncidentCase (
                ComplaintText,
                ImmediateAction,
                TakenAction,
                FeedbackRecievedDate,
                PatientName,
                IssuingOrgUnitID,
                isINPatient,
                ClinicalRiskTypeID,
                FeedbackIntentTypeID,
                BuildingID,
                DomainID,
                CategoryID,
                SubCategoryID,
                ClassificationID,
                SeverityID,
                StageID,
                HarmLevelID,
                SourceID,
                CaseStatusID,
                ExplanationStatusID,
                CreatedByUserID,
                CreatedAt
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, GETDATE())
        """
        
        cursor.execute(insert_query, (
            payload.get('complaint_text'),
            payload.get('immediate_action'),
            payload.get('taken_action'),
            payload.get('feedback_received_date'),
            payload.get('patient_name'),
            payload.get('issuing_department_id'),
            0,  # isINPatient default to outpatient
            payload.get('clinical_risk_type_id', 1),
            payload.get('feedback_intent_type_id', 1),
            None,  # BuildingID
            payload.get('domain_id'),
            payload.get('category_id'),
            payload.get('subcategory_id'),
            payload.get('classification_id'),
            payload.get('severity_id'),
            payload.get('stage_id'),
            payload.get('harm_id'),
            payload.get('source_id', 1),
            3,  # CaseStatusID = Closed
            4,  # ExplanationStatusID = No Explanation Needed
            1   # CreatedByUserID (system import)
        ))
        
        conn.commit()
        
        return {'success': True}
    
    except Exception as e:
        conn.rollback()
        return {'success': False, 'error': str(e)}
