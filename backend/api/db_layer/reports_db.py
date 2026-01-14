"""
Database layer for reporting queries.
Handles all SQL queries for complaint aggregation, filtering, and statistics.
"""

import pyodbc
from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple

def get_connection():
    """Get database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


# =============================================
# B1: FETCH FILTERED COMPLAINTS (DETAILED MODE)
# =============================================

def get_filtered_complaints(
    year: int,
    month: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    building_id: Optional[int] = None,
    idara_id: Optional[int] = None,
    dayra_id: Optional[int] = None,
    qism_id: Optional[int] = None,
    domain_id: Optional[int] = None,
    category_id: Optional[int] = None,
    severity_id: Optional[int] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch paginated filtered complaints with all detail fields.
    
    Returns:
        Tuple of (complaints_list, total_record_count)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build date range
    if start_date and end_date:
        date_filter = f"AND ic.FeedbackRecievedDate BETWEEN '{start_date}' AND '{end_date}'"
    elif month:
        date_filter = f"AND YEAR(ic.FeedbackRecievedDate) = {year} AND MONTH(ic.FeedbackRecievedDate) = {month}"
    else:
        date_filter = f"AND YEAR(ic.FeedbackRecievedDate) = {year}"
    
    # Build WHERE clause
    where_parts = [date_filter]
    
    if building_id:
        where_parts.append(f"AND ic.BuildingID = {building_id}")
    if idara_id:
        where_parts.append(f"AND org_unit.ParentID = {idara_id}")
    if dayra_id:
        where_parts.append(f"AND tdep.DepartmentID = {dayra_id}")
    if qism_id:
        where_parts.append(f"AND org_unit.UniqueID = {qism_id}")
    if domain_id:
        where_parts.append(f"AND ic.DomainID = {domain_id}")
    if category_id:
        where_parts.append(f"AND ic.CategoryID = {category_id}")
    if severity_id:
        where_parts.append(f"AND ic.SeverityID = {severity_id}")
    if status:
        if status.lower() == "open":
            where_parts.append("AND ic.CaseStatusID != 3")
        elif status.lower() == "closed":
            where_parts.append("AND ic.CaseStatusID = 3")
    
    where_clause = " ".join(where_parts)
    
    # Count total records
    count_query = f"""
    SELECT COUNT(DISTINCT ic.IncidentRequestCaseID) as total
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit org_unit ON ic.IssuingOrgUnitID = org_unit.UniqueID
    LEFT JOIN dbo.APP_IncidentCaseTargetDepartment tdep ON ic.IncidentRequestCaseID = tdep.IncidentRequestCaseID
    WHERE 1=1 {where_clause}
    """
    
    cursor.execute(count_query)
    total_records = cursor.fetchone()[0]
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Main query - FULL COMPLAINT DTO (matches TableView/Single Complaint endpoint)
    query = f"""
    SELECT DISTINCT
        ic.IncidentRequestCaseID as id,
        ic.ComplaintText as complaint_text,
        ic.ImmediateAction as immediate_action,
        ic.TakenAction as taken_action,
        ic.FeedbackRecievedDate as received_date,
        ic.PatientName as patient_name,
        ic.CreatedAt as created_at,
        ic.CreatedByUserID as created_by_user_id,
        ic.isINPatient as is_inpatient,
        
        -- Issuing organizational unit
        ic.IssuingOrgUnitID as issuing_org_unit_id,
        org_unit.Name as issuing_org_unit_name,
        
        -- Organizational hierarchy (3 levels)
        COALESCE(sec_unit.Name, '—') as section_name,
        COALESCE(dept_unit.Name, '—') as department_name,
        COALESCE(admin_unit.Name, '—') as administration_name,
        
        -- Domain
        ic.DomainID as domain_id,
        domain.DomainName as domain_name,
        
        -- Category
        ic.CategoryID as category_id,
        category.CategoryName as category_name,
        
        -- SubCategory
        ic.SubCategoryID as subcategory_id,
        subcategory.SubCategoryName as subcategory_name,
        
        -- Classification
        ic.ClassificationID as classification_id,
        classification.Classification_AR as classification_name,
        classification.Classification_EN as classification_name_en,
        
        -- Severity
        ic.SeverityID as severity_id,
        severity.SeverityName as severity_name,
        
        -- Stage
        ic.StageID as stage_id,
        stage.StageName as stage_name,
        
        -- Harm level
        ic.HarmLevelID as harm_level_id,
        harm.HarmLevel as harm_level,
        
        -- Case Status
        ic.CaseStatusID as case_status_id,
        status.Name as status_name,
        
        -- Building
        ic.BuildingID as building_id,
        building.BuildingName as building_name,
        
        -- Risk and Intent Types
        ic.ClinicalRiskTypeID as clinical_risk_type_id,
        clinical_risk.Name as clinical_risk_type_name,
        ic.FeedbackIntentTypeID as feedback_intent_type_id,
        feedback_intent.NameEn as feedback_intent_type_name,
        
        -- Source
        ic.SourceID as source_id,
        source.SourceName as source_name,
        
        -- Explanation Status
        ic.ExplanationStatusID as explanation_status_id,
        explanation_status.StatusName as explanation_status_name
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit org_unit ON ic.IssuingOrgUnitID = org_unit.UniqueID
    LEFT JOIN dbo.APP_IncidentCaseTargetDepartment tdep ON ic.IncidentRequestCaseID = tdep.IncidentRequestCaseID
    
    -- Organizational hierarchy joins (3 levels)
    LEFT JOIN dbo.AdminsrationUnit sec_unit ON ic.IssuingOrgUnitID = sec_unit.UniqueID  -- Section (leaf)
    LEFT JOIN dbo.AdminsrationUnit dept_unit ON sec_unit.ParentID = dept_unit.UniqueID   -- Department (middle)
    LEFT JOIN dbo.AdminsrationUnit admin_unit ON dept_unit.ParentID = admin_unit.UniqueID -- Administration (top)
    
    LEFT JOIN dbo.APP_LOOKUP_DOMAIN domain ON ic.DomainID = domain.DomainID
    LEFT JOIN dbo.APP_LOOKUP_CATEGORY category ON ic.CategoryID = category.CategoryID
    LEFT JOIN dbo.APP_LOOKUP_SUBCATEGORY subcategory ON ic.SubCategoryID = subcategory.SubCategoryID
    LEFT JOIN dbo.APP_LOOKUP_CLASSIFICATION classification ON ic.ClassificationID = classification.ClassificationID
    LEFT JOIN dbo.APP_LOOKUP_SEVERITY severity ON ic.SeverityID = severity.SeverityID
    LEFT JOIN dbo.APP_LOOKUP_CASE_STAGE stage ON ic.StageID = stage.StageID
    LEFT JOIN dbo.APP_LOOKUP_HARM_LEVEL harm ON ic.HarmLevelID = harm.HarmID
    LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS status ON ic.CaseStatusID = status.CaseStatusID
    LEFT JOIN dbo.APP_LOOKUP_BUILDING building ON ic.BuildingID = building.BuildingID
    LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE clinical_risk ON ic.ClinicalRiskTypeID = clinical_risk.ClinicalRiskTypeID
    LEFT JOIN dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE feedback_intent ON ic.FeedbackIntentTypeID = feedback_intent.FeedbackIntentTypeID
    LEFT JOIN dbo.APP_LOOKUP_SOURCE source ON ic.SourceID = source.SourceID
    LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS explanation_status ON ic.ExplanationStatusID = explanation_status.StatusID
    WHERE 1=1 {where_clause}
    ORDER BY ic.FeedbackRecievedDate DESC
    OFFSET {offset} ROWS FETCH NEXT {page_size} ROWS ONLY
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    
    complaints = []
    for row in rows:
        complaint = dict(zip(columns, row))
        
        # Format dates
        if complaint.get('received_date'):
            if isinstance(complaint['received_date'], datetime):
                complaint['received_date'] = complaint['received_date'].strftime('%Y-%m-%d')
            elif isinstance(complaint['received_date'], date):
                complaint['received_date'] = complaint['received_date'].isoformat()
        
        if complaint.get('created_at'):
            if isinstance(complaint['created_at'], datetime):
                complaint['created_at'] = complaint['created_at'].isoformat()
        
        # Fetch target departments for this complaint (same logic as single complaint endpoint)
        target_dept_query = """
            SELECT 
                td.DepartmentID as department_id,
                org.Name as department_name,
                td.IsPrimary as is_primary
            FROM dbo.APP_IncidentCaseTargetDepartment td
            LEFT JOIN dbo.AdminsrationUnit org ON td.DepartmentID = org.UniqueID
            WHERE td.IncidentRequestCaseID = ?
            ORDER BY td.IsPrimary DESC, td.DepartmentID
        """
        cursor.execute(target_dept_query, (complaint['id'],))
        target_dept_rows = cursor.fetchall()
        
        target_departments = []
        for dept_row in target_dept_rows:
            target_departments.append({
                'department_id': dept_row.department_id,
                'department_name': dept_row.department_name,
                'is_primary': bool(dept_row.is_primary)
            })
        
        complaint['target_departments'] = target_departments
        
        complaints.append(complaint)
    
    conn.close()
    return complaints, total_records


# =============================================
# B2: MONTHLY AGGREGATED STATISTICS
# =============================================

def get_monthly_statistics(
    year: int,
    month: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    building_id: Optional[int] = None,
    idara_id: Optional[int] = None,
    dayra_id: Optional[int] = None,
    qism_id: Optional[int] = None
) -> Dict[str, Any]:
    """Fetch aggregated monthly statistics."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build date range
    if start_date and end_date:
        date_filter = f"WHERE ic.FeedbackRecievedDate BETWEEN '{start_date}' AND '{end_date}'"
    elif month:
        date_filter = f"WHERE YEAR(ic.FeedbackRecievedDate) = {year} AND MONTH(ic.FeedbackRecievedDate) = {month}"
    else:
        date_filter = f"WHERE YEAR(ic.FeedbackRecievedDate) = {year}"
    
    # Build additional filters
    filter_parts = []
    if building_id:
        filter_parts.append(f"ic.BuildingID = {building_id}")
    if dayra_id:
        filter_parts.append(f"ic.IssuingOrgUnitID = {dayra_id}")
    if qism_id:
        filter_parts.append(f"ou.UniqueID = {qism_id}")
    
    additional_filter = " AND ".join(filter_parts)
    if additional_filter:
        date_filter += f" AND {additional_filter}"
    
    # Summary stats
    summary_query = f"""
    SELECT 
        COUNT(*) as total_complaints,
        SUM(CASE WHEN ic.CaseStatusID != 3 THEN 1 ELSE 0 END) as open_complaints,
        SUM(CASE WHEN ic.CaseStatusID = 3 THEN 1 ELSE 0 END) as closed_complaints,
        SUM(CASE WHEN ic.ClassificationID >= 78 THEN 1 ELSE 0 END) as red_flags_count,
        SUM(CASE WHEN ic.HarmLevelID = 5 THEN 1 ELSE 0 END) as never_events_count,
        AVG(CASE WHEN ic.CaseStatusID = 3 THEN DATEDIFF(DAY, ic.FeedbackRecievedDate, ic.CreatedAt) ELSE NULL END) as avg_closure_days
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit ou ON ic.IssuingOrgUnitID = ou.UniqueID
    {date_filter}
    """
    
    cursor.execute(summary_query)
    summary_row = cursor.fetchone()
    summary = {
        "total_complaints": summary_row[0] or 0,
        "open_complaints": summary_row[1] or 0,
        "closed_complaints": summary_row[2] or 0,
        "red_flags_count": summary_row[3] or 0,
        "never_events_count": summary_row[4] or 0,
        "avg_closure_days": float(summary_row[5]) if summary_row[5] else 0.0,
        "median_closure_days": 0.0
    }
    
    # By domain
    domain_query = f"""
    SELECT 
        ic.DomainID,
        COUNT(*) as count,
        ROUND(CAST(COUNT(*) AS FLOAT) / SUM(COUNT(*)) OVER () * 100, 1) as percentage
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit ou ON ic.IssuingOrgUnitID = ou.UniqueID
    {date_filter}
    GROUP BY ic.DomainID
    ORDER BY count DESC
    """
    
    cursor.execute(domain_query)
    domain_rows = cursor.fetchall()
    by_domain = []
    for row in domain_rows:
        by_domain.append({
            "domain_id": row[0],
            "domain_name": f"Domain {row[0]}",
            "domain_name_ar": f"المجال {row[0]}",
            "count": row[1],
            "percentage": row[2] if row[2] else 0.0
        })
    
    # By severity
    severity_query = f"""
    SELECT 
        ic.SeverityID,
        COUNT(*) as count
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit ou ON ic.IssuingOrgUnitID = ou.UniqueID
    {date_filter}
    GROUP BY ic.SeverityID
    ORDER BY ic.SeverityID
    """
    
    cursor.execute(severity_query)
    severity_rows = cursor.fetchall()
    by_severity = []
    for row in severity_rows:
        severity_name = "Medium"
        severity_name_ar = "متوسط"
        if row[0] == 1:
            severity_name = "Low"
            severity_name_ar = "منخفض"
        elif row[0] == 2:
            severity_name = "High"
            severity_name_ar = "عالي"
        
        by_severity.append({
            "severity_id": row[0],
            "severity_name": severity_name,
            "severity_name_ar": severity_name_ar,
            "count": row[1]
        })
    
    # By department
    dept_query = f"""
    SELECT 
        ic.IssuingOrgUnitID,
        COALESCE(ou.Name, 'Unknown') as dept_name,
        COUNT(*) as count
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit ou ON ic.IssuingOrgUnitID = ou.UniqueID
    {date_filter}
    GROUP BY ic.IssuingOrgUnitID, ou.Name
    ORDER BY count DESC
    """
    
    cursor.execute(dept_query)
    dept_rows = cursor.fetchall()
    by_department = []
    for row in dept_rows:
        by_department.append({
            "dayra_id": row[0],
            "dayra_name": row[1] or "Unknown",
            "dayra_name_ar": row[1] or "غير معروف",
            "count": row[2]
        })
    
    conn.close()
    
    return {
        "summary": summary,
        "by_domain": by_domain,
        "by_category": [],
        "by_severity": by_severity,
        "by_department": by_department
    }


# =============================================
# B3: SEASONAL HCAT ANALYSIS
# =============================================

def get_seasonal_hcat(
    year: int,
    start_date: date,
    end_date: date,
    threshold: int = 50,
    building_id: Optional[int] = None,
    idara_id: Optional[int] = None,
    dayra_id: Optional[int] = None
) -> Dict[str, Any]:
    """Fetch seasonal HCAT analysis with threshold evaluation."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build filters
    filter_parts = [f"ic.FeedbackRecievedDate BETWEEN '{start_date}' AND '{end_date}'"]
    
    if building_id:
        filter_parts.append(f"ic.BuildingID = {building_id}")
    if dayra_id:
        filter_parts.append(f"ic.IssuingOrgUnitID = {dayra_id}")
    
    where_clause = " AND ".join(filter_parts)
    
    # Total complaints in period
    total_query = f"""
    SELECT COUNT(*) FROM dbo.APP_IncidentCase ic
    WHERE {where_clause}
    """
    cursor.execute(total_query)
    total_complaints = cursor.fetchone()[0]
    
    # Domain analysis
    domain_query = f"""
    SELECT 
        ic.DomainID,
        COUNT(*) as complaint_count
    FROM dbo.APP_IncidentCase ic
    WHERE {where_clause}
    GROUP BY ic.DomainID
    ORDER BY complaint_count DESC
    """
    
    cursor.execute(domain_query)
    domain_rows = cursor.fetchall()
    
    domains = []
    for row in domain_rows:
        domain_id, complaint_count = row
        exceeds = complaint_count >= threshold
        ratio = complaint_count / threshold if threshold > 0 else 0.0
        
        # Get categories within this domain
        cat_query = f"""
        SELECT 
            ic.CategoryID,
            COUNT(*) as count
        FROM dbo.APP_IncidentCase ic
        WHERE {where_clause} AND ic.DomainID = {domain_id}
        GROUP BY ic.CategoryID
        ORDER BY count DESC
        """
        cursor.execute(cat_query)
        cat_rows = cursor.fetchall()
        
        categories = []
        for cat_row in cat_rows:
            cat_id, cat_count = cat_row
            cat_percentage = (cat_count / complaint_count * 100) if complaint_count > 0 else 0.0
            categories.append({
                "category_id": cat_id,
                "category_name": f"Category {cat_id}",
                "category_name_ar": f"الفئة {cat_id}",
                "count": cat_count,
                "percentage": round(cat_percentage, 1)
            })
        
        domains.append({
            "domain_id": domain_id,
            "domain_name": f"Domain {domain_id}",
            "domain_name_ar": f"المجال {domain_id}",
            "complaint_count": complaint_count,
            "exceeds_threshold": exceeds,
            "threshold_ratio": round(ratio, 2),
            "trend_direction": "stable",
            "categories": categories
        })
    
    conn.close()
    
    exceeding_count = sum(1 for d in domains if d["exceeds_threshold"])
    
    return {
        "total_complaints": total_complaints,
        "threshold_value": threshold,
        "domains": domains,
        "exceeding_count": exceeding_count,
        "within_threshold_count": len(domains) - exceeding_count
    }


# =============================================
# B4: BULK EXPORT SUMMARY (PER DEPARTMENT)
# =============================================

def get_bulk_summary(
    year: int,
    month: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    building_id: Optional[int] = None,
    idara_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetch department-level summaries for bulk export."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build date range
    if start_date and end_date:
        date_filter = f"ic.FeedbackRecievedDate BETWEEN '{start_date}' AND '{end_date}'"
    elif month:
        date_filter = f"YEAR(ic.FeedbackRecievedDate) = {year} AND MONTH(ic.FeedbackRecievedDate) = {month}"
    else:
        date_filter = f"YEAR(ic.FeedbackRecievedDate) = {year}"
    
    # Build additional filters
    filter_parts = [date_filter]
    if building_id:
        filter_parts.append(f"ic.BuildingID = {building_id}")
    
    where_clause = " AND ".join(filter_parts)
    
    # Department summaries
    dept_query = f"""
    SELECT 
        ic.IssuingOrgUnitID as dayra_id,
        COALESCE(ou.Name, 'Unknown') as dayra_name,
        COUNT(*) as total_complaints,
        SUM(CASE WHEN ic.CaseStatusID != 3 THEN 1 ELSE 0 END) as open_complaints,
        SUM(CASE WHEN ic.CaseStatusID = 3 THEN 1 ELSE 0 END) as closed_complaints,
        SUM(CASE WHEN ic.ClassificationID >= 78 THEN 1 ELSE 0 END) as red_flags_count,
        SUM(CASE WHEN ic.HarmLevelID = 5 THEN 1 ELSE 0 END) as never_events_count,
        TOP 1 ic.DomainID as top_domain_id
    FROM dbo.APP_IncidentCase ic
    LEFT JOIN dbo.AdminsrationUnit ou ON ic.IssuingOrgUnitID = ou.UniqueID
    WHERE {where_clause}
    GROUP BY ic.IssuingOrgUnitID, ou.Name
    ORDER BY total_complaints DESC
    """
    
    cursor.execute(dept_query)
    dept_rows = cursor.fetchall()
    
    departments = []
    for row in dept_rows:
        departments.append({
            "dayra_id": row[0],
            "dayra_name": row[1] or "Unknown",
            "dayra_name_ar": row[1] or "غير معروف",
            "total_complaints": row[2],
            "open_complaints": row[3] or 0,
            "closed_complaints": row[4] or 0,
            "red_flags_count": row[5] or 0,
            "never_events_count": row[6] or 0,
            "top_domain": f"Domain {row[7] or 'Unknown'}",
            "top_domain_ar": f"المجال {row[7] or 'غير معروف'}",
            "top_domain_count": 0
        })
    
    conn.close()
    return departments
