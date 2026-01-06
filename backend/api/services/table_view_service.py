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
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill

from ..db_layer.database import get_connection


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
        "CreatedAt", "PatientName"
    ]
    if sort_by not in valid_sort_fields:
        sort_by = "FeedbackRecievedDate"
    
    if sort_order.lower() not in ["asc", "desc"]:
        sort_order = "desc"
    
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
    
    # Build ORDER BY clause
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
            
            -- Classification
            c.ClassificationID as classification_id,
            
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
            
            -- Other fields
            c.ImmediateAction as immediate_action,
            c.TakenAction as taken_action,
            c.isINPatient as is_inpatient,
            c.SourceID as source_id,
            c.CreatedByUserID as created_by_user_id
            
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
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
                c.ClassificationName as name
                """ + (", COUNT(ic.IncidentRequestCaseID) as count" if include_counts else "") + """
            FROM APP_LOOKUP_CLASSIFICATION c
            """ + ("LEFT JOIN APP_IncidentCase ic ON c.ClassificationID = ic.ClassificationID" if include_counts else "") + """
            """ + ("GROUP BY c.ClassificationID, c.ClassificationName" if include_counts else "") + """
            ORDER BY c.ClassificationName
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
            
            -- Classification
            c.ClassificationID as classification_id,
            
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
            
            -- Risk and Intent Types
            c.ClinicalRiskTypeID as clinical_risk_type_id,
            c.FeedbackIntentTypeID as feedback_intent_type_id,
            
            -- Source
            c.SourceID as source_id
        FROM dbo.APP_IncidentCase c
        LEFT JOIN AdminsrationUnit org_unit ON c.IssuingOrgUnitID = org_unit.UniqueID
        LEFT JOIN APP_LOOKUP_DOMAIN domain ON c.DomainID = domain.DomainID
        LEFT JOIN APP_LOOKUP_CATEGORY category ON c.CategoryID = category.CategoryID
        LEFT JOIN APP_LOOKUP_SEVERITY severity ON c.SeverityID = severity.SeverityID
        LEFT JOIN APP_LOOKUP_CASE_STAGE stage ON c.StageID = stage.StageID
        LEFT JOIN APP_LOOKUP_HARM_LEVEL harm ON c.HarmLevelID = harm.HarmID
        LEFT JOIN APP_LOOKUP_CASE_STATUS status ON c.CaseStatusID = status.CaseStatusID
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
            N'' as source_name,
            CASE 
                WHEN c.FeedbackIntentTypeID = 1 THEN N'شكوى'
                WHEN c.FeedbackIntentTypeID = 2 THEN N'ملاحظة'
                WHEN c.FeedbackIntentTypeID = 3 THEN N'اقتراح'
                WHEN c.FeedbackIntentTypeID = 4 THEN N'استفسار'
                ELSE N'غير محدد'
            END as feedback_type,
            domain.DomainName as domain_name,
            category.CategoryName as category_name,
            CAST(c.SubCategoryID AS NVARCHAR(50)) as subcategory_name,
            CAST(c.ClassificationID AS NVARCHAR(50)) as classification_name,
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
            'source_name': 'المصدر 1',
            'feedback_type': 'النوع (Feedback Type)',
            'domain_name': 'Domain',
            'category_name': 'Category',
            'subcategory_name': 'SubCategory',
            'classification_name': 'New-Classification in Arabic',
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
