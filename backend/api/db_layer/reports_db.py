"""
Database layer for reporting queries.
Handles all SQL queries for complaint aggregation, filtering, and statistics.
"""

from datetime import datetime, date
from typing import Dict, List, Any, Optional, Tuple
from core.database import get_connection

def get_org_unit_descendants(unit_id: int) -> List[int]:
    """
    Get all descendant organizational unit IDs (including the unit itself) using recursive CTE.
    This works for any depth of organizational hierarchy.
    Handles self-referencing root nodes properly to avoid infinite loops.
    
    Args:
        unit_id: The organizational unit ID to get descendants for
        
    Returns:
        List of all descendant unit IDs including the unit itself
    """
    if not unit_id:
        return []
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # Recursive CTE to find all descendants with loop prevention
    query = """
    WITH OrgTree AS (
        -- Anchor: Start with the selected unit
        SELECT UniqueID, 0 as level
        FROM dbo.AdminsrationUnit 
        WHERE UniqueID = ?
        
        UNION ALL
        
        -- Recursive: Find all children (prevent self-reference loops)
        SELECT child.UniqueID, parent.level + 1
        FROM dbo.AdminsrationUnit child
        INNER JOIN OrgTree parent ON child.ParentID = parent.UniqueID
        WHERE child.UniqueID != child.ParentID  -- Avoid infinite loops from self-references
        AND parent.level < 10  -- Additional safety limit
    )
    SELECT DISTINCT UniqueID FROM OrgTree
    """
    
    try:
        cursor.execute(query, (unit_id,))
        descendants = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()
        return descendants
    except Exception as e:
        cursor.close()
        conn.close()
        # Fallback: return just the unit itself if recursive query fails
        return [unit_id]


def debug_expand_org_units(unit_ids: List[int]) -> List[int]:
    """
    Debug helper function to expand a list of org unit IDs into all their descendants.
    
    Args:
        unit_ids: List of organizational unit IDs to expand
        
    Returns:
        List of all expanded unit IDs (including originals and all descendants)
    """
    expanded = set()
    for uid in unit_ids:
        expanded.update(get_org_unit_descendants(uid))
    return list(expanded)


def build_org_filter_condition(building_id: Optional[int] = None, idara_id: Optional[int] = None, 
                              dayra_id: Optional[int] = None, qism_id: Optional[int] = None) -> str:
    """
    Build a tree-aware organizational filtering condition for target departments.
    
    CRITICAL LOGIC:
    - When you filter by Administration X, get ALL complaints where ANY target department 
      belongs to Administration X or ANY of its descendants (departments, sections)
    - Same logic applies for Department and Section filtering
    - Works regardless of primary/non-primary status
    - Uses recursive tree expansion to include entire subtree
    
    Args:
        building_id: Hospital/Building ID
        idara_id: Administration ID (single int) - will expand to include all departments/sections under it
        dayra_id: Department ID (single int) - will expand to include all sections under it
        qism_id: Section ID (single int) - leaf node, no expansion needed
        
    Returns:
        SQL WHERE condition string for target department filtering
    
    Example:
        If Administration 3 has:
            - Department 28 → Sections [43, 44, 45]
            - Department 24 → Sections [46, 47]
        
        Filter by idara_id=3 will find complaints where ANY target department is in:
        [3, 28, 24, 43, 44, 45, 46, 47]
    """
    # Collect ALL selected organizational unit IDs (supporting multiple levels)
    selected_unit_ids = []
    
    # Priority order: Section > Department > Administration
    # If you specify Section, it takes precedence (most specific)
    if qism_id:
        if isinstance(qism_id, list):
            selected_unit_ids.extend(qism_id)
        else:
            selected_unit_ids.append(qism_id)
        print(f"[ORG FILTER] Section filter: {qism_id}")
    
    if dayra_id:
        if isinstance(dayra_id, list):
            selected_unit_ids.extend(dayra_id)
        else:
            selected_unit_ids.append(dayra_id)
        print(f"[ORG FILTER] Department filter: {dayra_id}")
            
    if idara_id:
        if isinstance(idara_id, list):
            selected_unit_ids.extend(idara_id)
        else:
            selected_unit_ids.append(idara_id)
        print(f"[ORG FILTER] Administration filter: {idara_id}")
    
    if building_id:
        # For building/hospital level, return all (no org filtering)
        print(f"[ORG FILTER] Hospital/Building level - no org filter applied")
        return "1=1"
        
    if not selected_unit_ids:
        print("[ORG FILTER] No org filter specified - returning all")
        return "1=1"  # No org filtering
    
    # CRITICAL: Expand each selected unit to include ALL its descendants
    # Example: Administration 3 → expands to [3, 28, 24, 43, 44, 45, 46, 47]
    expanded_org_unit_ids = debug_expand_org_units(selected_unit_ids)
    
    print(f"[ORG FILTER] Input IDs: {selected_unit_ids}")
    print(f"[ORG FILTER] Expanded to include descendants: {expanded_org_unit_ids}")
    
    if not expanded_org_unit_ids:
        print("[ORG FILTER] WARNING: No descendants found - returning no results")
        return "1=0"  # No results if no descendants found
    
    # Build the filter: Find complaints where ANY target department is in the expanded list
    # This checks APP_IncidentCaseTargetDepartment table (regardless of IsPrimary status)
    id_list = ",".join(str(id) for id in expanded_org_unit_ids)
    
    filter_condition = f"""EXISTS (
        SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment td_filter 
        WHERE td_filter.IncidentRequestCaseID = ic.IncidentRequestCaseID 
        AND td_filter.DepartmentID IN ({id_list})
    )"""
    
    print(f"[ORG FILTER] Generated filter for {len(expanded_org_unit_ids)} unit IDs")
    
    return filter_condition


# =============================================
# B1: FETCH FILTERED COMPLAINTS (DETAILED MODE)
# =============================================

def get_filtered_complaints(
    year: int,
    month: Optional[int] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    allowed_unit_ids: Optional[List[int]] = None,
    target_unit_ids: Optional[List[int]] = None,
    domain_id: Optional[int] = None,
    category_id: Optional[int] = None,
    severity_id: Optional[int] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Fetch paginated filtered complaints with all detail fields.
    
    Phase 2.5.7: Uses allowed_unit_ids from scope engine (server authority).
    
    Args:
        allowed_unit_ids: List of org unit IDs from current_user.allowed_unit_ids (security boundary)
        target_unit_ids: Optional list of target department IDs to filter by.
                         When set, only returns cases that TARGET these sections
                         (via APP_IncidentCaseTargetDepartment). Used by multi-export
                         to generate per-section files.
        
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
    
    # Phase 2.5.7: Security boundary — filter by allowed_unit_ids (scope engine authority)
    if allowed_unit_ids:
        placeholders = ','.join(str(uid) for uid in allowed_unit_ids)
        where_parts.append(f"AND ic.IssuingOrgUnitID IN ({placeholders})")
    else:
        # No allowed units - return empty result (fail-safe)
        where_parts.append("AND 1=0")
    
    # Target department filter — narrows to cases TARGETING specific sections
    # Used by multi-export to isolate per-section data
    if target_unit_ids:
        target_placeholders = ','.join(str(uid) for uid in target_unit_ids)
        where_parts.append(f"""AND EXISTS (
            SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment td_filter
            WHERE td_filter.IncidentRequestCaseID = ic.IncidentRequestCaseID
            AND td_filter.DepartmentID IN ({target_placeholders})
        )""")
    
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
    
    # Count total records - count DISTINCT complaints only
    count_query = f"""
    SELECT COUNT(DISTINCT ic.IncidentRequestCaseID) as total
    FROM dbo.APP_IncidentCase ic
    WHERE 1=1 {where_clause}
    """
    
    cursor.execute(count_query)
    total_records = cursor.fetchone()[0]
    
    # Calculate offset
    offset = (page - 1) * page_size
    
    # Main query - one record per COMPLAINT (not per target department)
    # Target departments are fetched separately to avoid duplicates
    query = f"""
    SELECT
        ic.IncidentRequestCaseID as id,
        ic.ComplaintText as complaint_text,
        ic.ImmediateAction as immediate_action,
        ic.TakenAction as taken_action,
        ic.FeedbackRecievedDate as received_date,
        ic.PatientName as patient_name,
        ic.CreatedAt as created_at,
        ic.CreatedByUserID as created_by_user_id,
        ic.isINPatient as is_inpatient,
        
        -- Issuing organizational unit (the section that received/created the complaint)
        ic.IssuingOrgUnitID as issuing_org_unit_id,
        issuing_section.Name as issuing_org_unit_name,
        
        -- No target department in main query (fetched separately)
        NULL as target_department_id,
        NULL as target_department_name,
        
        -- Issuing organizational hierarchy (Section -> Department -> Administration)
        issuing_section.Name as section_name,
        issuing_dept.Name as department_name,
        issuing_admin.Name as administration_name,
        
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
    
    -- Join issuing organizational unit hierarchy (Section -> Department -> Administration)
    LEFT JOIN dbo.AdminsrationUnit issuing_section ON ic.IssuingOrgUnitID = issuing_section.UniqueID
    LEFT JOIN dbo.AdminsrationUnit issuing_dept ON issuing_section.ParentID = issuing_dept.UniqueID
    LEFT JOIN dbo.AdminsrationUnit issuing_admin ON issuing_dept.ParentID = issuing_admin.UniqueID
    
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
            ORDER BY td.DepartmentID
        """
        cursor.execute(target_dept_query, (complaint['id'],))
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
    allowed_unit_ids: Optional[List[int]] = None,
    target_unit_ids: Optional[List[int]] = None,
    group_by: str = "section"
) -> Dict[str, Any]:
    """
    Fetch aggregated monthly statistics.
    
    Phase 2.5.7: Uses allowed_unit_ids from scope engine (server authority).
    
    Args:
        allowed_unit_ids: Security boundary (IssuingOrgUnitID filter)
        target_unit_ids: Optional target department filter (via APP_IncidentCaseTargetDepartment)
        group_by: Aggregation level for by_department breakdown.
                  "section" (default) - group by target department (section level)
                  "department" - roll up to parent department (Type=325)
                  "administration" - roll up to grandparent administration (Type=323)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build date range
    if start_date and end_date:
        date_filter = f"WHERE ic.FeedbackRecievedDate BETWEEN '{start_date}' AND '{end_date}'"
    elif month:
        date_filter = f"WHERE YEAR(ic.FeedbackRecievedDate) = {year} AND MONTH(ic.FeedbackRecievedDate) = {month}"
    else:
        date_filter = f"WHERE YEAR(ic.FeedbackRecievedDate) = {year}"
    
    # Phase 2.5.7: Security boundary — filter by allowed_unit_ids
    if allowed_unit_ids:
        placeholders = ','.join(str(uid) for uid in allowed_unit_ids)
        date_filter += f" AND ic.IssuingOrgUnitID IN ({placeholders})"
    else:
        # No allowed units - return empty result (fail-safe)
        date_filter += " AND 1=0"
    
    # Target department filter — narrows to cases TARGETING specific sections
    if target_unit_ids:
        target_placeholders = ','.join(str(uid) for uid in target_unit_ids)
        date_filter += f""" AND EXISTS (
            SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment td_filter
            WHERE td_filter.IncidentRequestCaseID = ic.IncidentRequestCaseID
            AND td_filter.DepartmentID IN ({target_placeholders})
        )"""
    
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
    
    # By target department - with group_by level support
    # Hierarchy: Section (Type=324) → Department (Type=325) → Administration (Type=323)
    # Some sections parent directly to an Administration (no intermediate Department)
    if group_by == "administration":
        # Roll up to the administration level (grandparent or parent if section→admin directly)
        # Walk up the tree: target dept → parent → grandparent, pick the first Type=323 node
        dept_query = f"""
        SELECT 
            admin_unit.UniqueID as GroupID,
            COALESCE(admin_unit.Name, 'Unknown') as group_name,
            COUNT(td.IncidentRequestCaseID) as count
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseTargetDepartment td ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        LEFT JOIN dbo.AdminsrationUnit sec_unit ON td.DepartmentID = sec_unit.UniqueID
        LEFT JOIN dbo.AdminsrationUnit dept_unit ON sec_unit.ParentID = dept_unit.UniqueID
        LEFT JOIN dbo.AdminsrationUnit admin_from_dept ON dept_unit.ParentID = admin_from_dept.UniqueID
        CROSS APPLY (
            SELECT CASE 
                WHEN sec_unit.Type = 323 THEN sec_unit.UniqueID
                WHEN dept_unit.Type = 323 THEN dept_unit.UniqueID
                WHEN admin_from_dept.Type = 323 THEN admin_from_dept.UniqueID
                ELSE COALESCE(dept_unit.UniqueID, sec_unit.UniqueID)
            END AS UniqueID,
            CASE 
                WHEN sec_unit.Type = 323 THEN sec_unit.Name
                WHEN dept_unit.Type = 323 THEN dept_unit.Name
                WHEN admin_from_dept.Type = 323 THEN admin_from_dept.Name
                ELSE COALESCE(dept_unit.Name, sec_unit.Name)
            END AS Name
        ) admin_unit
        {date_filter}
        GROUP BY admin_unit.UniqueID, admin_unit.Name
        ORDER BY count DESC
        """
    elif group_by == "department":
        # Roll up to the department level (parent of section)
        # If parent is an Administration (Type=323), use it as the "department" level
        dept_query = f"""
        SELECT 
            parent_unit.UniqueID as GroupID,
            COALESCE(parent_unit.Name, 'Unknown') as group_name,
            COUNT(td.IncidentRequestCaseID) as count
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseTargetDepartment td ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        LEFT JOIN dbo.AdminsrationUnit sec_unit ON td.DepartmentID = sec_unit.UniqueID
        LEFT JOIN dbo.AdminsrationUnit parent_unit ON sec_unit.ParentID = parent_unit.UniqueID
        {date_filter}
        GROUP BY parent_unit.UniqueID, parent_unit.Name
        ORDER BY count DESC
        """
    else:
        # Default: group by section (target department directly)
        dept_query = f"""
        SELECT 
            td.DepartmentID,
            COALESCE(ou.Name, 'Unknown') as dept_name,
            COUNT(td.IncidentRequestCaseID) as count
        FROM dbo.APP_IncidentCase ic
        INNER JOIN dbo.APP_IncidentCaseTargetDepartment td ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        LEFT JOIN dbo.AdminsrationUnit ou ON td.DepartmentID = ou.UniqueID
        {date_filter}
        GROUP BY td.DepartmentID, ou.Name
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
    
    # Generic tree-aware organizational filtering
    org_filter = build_org_filter_condition(building_id, idara_id, dayra_id, None)  # qism_id not used in seasonal
    if org_filter and org_filter != "1=1":
        filter_parts.append(org_filter)
    
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
    
    # Department summaries (based on target departments - count records, not distinct complaints)
    dept_query = f"""
    SELECT 
        td.DepartmentID as dayra_id,
        COALESCE(ou.Name, 'Unknown') as dayra_name,
        COUNT(td.IncidentRequestCaseID) as total_complaints,
        SUM(CASE WHEN ic.CaseStatusID != 3 THEN 1 ELSE 0 END) as open_complaints,
        SUM(CASE WHEN ic.CaseStatusID = 3 THEN 1 ELSE 0 END) as closed_complaints,
        SUM(CASE WHEN ic.ClassificationID >= 78 THEN 1 ELSE 0 END) as red_flags_count,
        SUM(CASE WHEN ic.HarmLevelID = 5 THEN 1 ELSE 0 END) as never_events_count,
        MAX(ic.DomainID) as top_domain_id
    FROM dbo.APP_IncidentCase ic
    INNER JOIN dbo.APP_IncidentCaseTargetDepartment td ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
    LEFT JOIN dbo.AdminsrationUnit ou ON td.DepartmentID = ou.UniqueID
    WHERE {where_clause}
    GROUP BY td.DepartmentID, ou.Name
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
