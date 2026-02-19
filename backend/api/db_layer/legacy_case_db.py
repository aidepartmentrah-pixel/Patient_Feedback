"""
PHASE K — Legacy Case DB Layer

Database layer functions for reading LEGACY case data for migration.

LEGACY TABLES (source for migration):
- IncidentRequestCase — Main legacy case table
- IncidentRequestCaseAction — Action history
- IncidentRequest — Parent request with requester note

NEW TABLE (target for migration):
- APP_IncidentCase — Current production table

This module provides read-only operations to retrieve legacy data
and transform it into the new format with 3 text fields:
1. complaint_content = Case Description + Requester Note
2. immediate_action = First action's fields
3. actions_taken = All remaining actions' fields

NO WRITES - Read-only legacy data access only.
"""

from core.database import get_connection


# =============================================================================
# HELPER FUNCTIONS — Text Field Building
# =============================================================================

def _join_with_double_newline(parts: list) -> str:
    """Join non-empty parts with double newline separator."""
    return "\n\n".join([p for p in parts if p and p.strip()])


def _join_with_single_newline(parts: list) -> str:
    """Join non-empty parts with single newline separator."""
    return "\n".join([p for p in parts if p and p.strip()])


def _build_complaint_content(case_description: str, requester_note: str) -> str:
    """
    Build complaint_content from legacy data.
    
    Sources:
    - IncidentRequestCase.Description
    - IncidentRequest.Note
    """
    parts = []
    
    if case_description and case_description.strip():
        parts.append("[Case Description]\n" + case_description.strip())
    
    if requester_note and requester_note.strip():
        parts.append("[Requester Note]\n" + requester_note.strip())
    
    return _join_with_double_newline(parts)


def _build_immediate_action(first_action: dict) -> str:
    """
    Build immediate_action from FIRST action row.
    
    Sources from first action:
    - Description
    - SectionNote
    - SelectionNote
    - ProblemReason
    """
    if not first_action:
        return ""
    
    parts = []
    
    if first_action.get("description") and first_action["description"].strip():
        parts.append("[Action Description]\n" + first_action["description"].strip())
    
    if first_action.get("section_note") and first_action["section_note"].strip():
        parts.append("[Section Note]\n" + first_action["section_note"].strip())
    
    if first_action.get("selection_note") and first_action["selection_note"].strip():
        parts.append("[Selection Note]\n" + first_action["selection_note"].strip())
    
    if first_action.get("problem_reason") and first_action["problem_reason"].strip():
        parts.append("[Problem Reason]\n" + first_action["problem_reason"].strip())
    
    return _join_with_double_newline(parts)


def _build_actions_taken(remaining_actions: list) -> str:
    """
    Build actions_taken from ALL remaining action rows (except first).
    
    Per action row, combines:
    - Description
    - Note
    - DepartmentNote
    - SectionNote
    - GoverningPolicies
    """
    if not remaining_actions:
        return ""
    
    action_blocks = []
    
    for action in remaining_actions:
        row_parts = []
        
        # Header with timestamp
        timestamp = action.get("date_created", "Unknown Date")
        if hasattr(timestamp, 'strftime'):
            timestamp = timestamp.strftime('%Y-%m-%d %H:%M')
        row_parts.append(f"[Action — {timestamp}]")
        
        if action.get("description") and action["description"].strip():
            row_parts.append("Description:\n" + action["description"].strip())
        
        if action.get("note") and action["note"].strip():
            row_parts.append("Note:\n" + action["note"].strip())
        
        if action.get("department_note") and action["department_note"].strip():
            row_parts.append("Department Note:\n" + action["department_note"].strip())
        
        if action.get("section_note") and action["section_note"].strip():
            row_parts.append("Section Note:\n" + action["section_note"].strip())
        
        if action.get("governing_policies") and action["governing_policies"].strip():
            row_parts.append("Policies:\n" + action["governing_policies"].strip())
        
        if len(row_parts) > 1:  # More than just header
            action_blocks.append(_join_with_single_newline(row_parts))
    
    return _join_with_double_newline(action_blocks)


# =============================================================================
# LIST LEGACY CASES (Paginated)
# =============================================================================

def list_legacy_cases_paged(page: int = 1, page_size: int = 50) -> dict:
    """
    Retrieve paginated list of legacy cases from IncidentRequestCase table.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
        
    Returns:
        dict: {
            "cases": [...],
            "total": int
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate offset
        offset = (page - 1) * page_size
        
        # Query legacy cases with parent request data
        cursor.execute("""
            SELECT 
                irc.UniqueID AS LegacyCaseID,
                irc.Description AS CaseDescription,
                ir.Note AS RequesterNote,
                ir.PatientName,
                ir.DateAndTimeRecieved AS FeedbackDate,
                irc.IncidentRequestCaseStatusID AS StatusID,
                irc.DateAndTimeCreated AS CreatedAt,
                CASE WHEN dm.MapID IS NOT NULL THEN 1 ELSE 0 END AS Migrated,
                dept.Name AS DepartmentName,
                sect.Name AS SectionName
            FROM dbo.IncidentRequestCase irc
            INNER JOIN dbo.IncidentRequest ir ON irc.IncidentRequestID = ir.UniqueID
            LEFT JOIN dbo.APP_DataMigration_Map dm ON irc.UniqueID = dm.legacy_case_id
            LEFT JOIN dbo.AdminsrationUnit dept ON irc.DepartmentID = dept.UniqueID
            LEFT JOIN dbo.AdminsrationUnit sect ON irc.SectionID = sect.UniqueID
            ORDER BY irc.UniqueID DESC
            OFFSET ? ROWS
            FETCH NEXT ? ROWS ONLY
        """, offset, page_size)
        
        rows = cursor.fetchall()
        
        # Query total count
        cursor.execute("SELECT COUNT(*) FROM dbo.IncidentRequestCase")
        total = cursor.fetchone()[0]
        
        cases = []
        for row in rows:
            # Build complaint_content preview from Description + Note
            case_desc = row[1] or ''
            req_note = row[2] or ''
            complaint_content = _build_complaint_content(case_desc, req_note)
            
            # Create preview from first 150 chars
            preview = complaint_content[:150] + '...' if len(complaint_content) > 150 else complaint_content
            
            # Use department or section name
            org_name = row[8] or row[9] or 'N/A'
            
            cases.append({
                "legacy_case_id": row[0],
                "id": row[0],  # Alias for frontend
                "complaint_text": complaint_content,
                "complaint_content": complaint_content,
                "preview": preview,
                "short_preview_text": preview,
                "patient_name": row[3],
                "feedback_received_date": row[4],
                "feedback_date": row[4],
                "case_status_id": row[5],
                "created_at": row[6],
                "migrated": bool(row[7]),
                "issuing_org_name": org_name,
                "department_name": org_name
            })
        
        return {
            "cases": cases,
            "total": total
        }
        
    except Exception as e:
        raise Exception("Failed to list legacy cases: " + str(e))
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_legacy_case_by_id(legacy_case_id: int) -> dict:
    """
    Retrieve detailed legacy case record from IncidentRequestCase table.
    
    Builds the 3 text fields according to migration mapping spec:
    1. complaint_content = Description + Requester Note
    2. immediate_action = First action's fields
    3. actions_taken = All remaining actions' fields
    
    Args:
        legacy_case_id: Legacy case ID from IncidentRequestCase.UniqueID
        
    Returns:
        dict: Full case record with built text fields
        None: If case not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # =====================================================================
        # QUERY 1: Main case data from IncidentRequestCase + IncidentRequest
        # =====================================================================
        cursor.execute("""
            SELECT 
                irc.UniqueID AS LegacyCaseID,
                irc.Description AS CaseDescription,
                irc.Note AS CaseNote,
                ir.Note AS RequesterNote,
                ir.PatientName,
                ir.DateAndTimeRecieved AS FeedbackDate,
                ir.IsInPatient,
                irc.DateAndTimeCreated AS CreatedAt,
                irc.IncidentRequestCaseStatusID AS StatusID,
                irc.DepartmentID,
                irc.SectionID,
                irc.DoctorID,
                irc.IncidentCaseCategoryID AS CategoryID,
                irc.IncidentCaseSubCategoryID AS SubCategoryID,
                irc.CaseBuilding AS Building,
                CASE WHEN dm.MapID IS NOT NULL THEN 1 ELSE 0 END AS Migrated,
                dm.new_case_id AS MigratedCaseID,
                dept.Name AS DepartmentName,
                sect.Name AS SectionName
            FROM dbo.IncidentRequestCase irc
            INNER JOIN dbo.IncidentRequest ir ON irc.IncidentRequestID = ir.UniqueID
            LEFT JOIN dbo.APP_DataMigration_Map dm ON irc.UniqueID = dm.legacy_case_id
            LEFT JOIN dbo.AdminsrationUnit dept ON irc.DepartmentID = dept.UniqueID
            LEFT JOIN dbo.AdminsrationUnit sect ON irc.SectionID = sect.UniqueID
            WHERE irc.UniqueID = ?
        """, legacy_case_id)
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Extract main case data
        legacy_case_id_val = row[0]
        case_description = row[1] or ''
        case_note = row[2] or ''
        requester_note = row[3] or ''
        patient_name = row[4]
        feedback_date = row[5]
        is_inpatient = row[6]
        created_at = row[7]
        status_id = row[8]
        department_id = row[9]
        section_id = row[10]
        doctor_id = row[11]
        category_id = row[12]
        subcategory_id = row[13]
        building = row[14]
        migrated = bool(row[15])
        migrated_case_id = row[16]
        department_name = row[17] or ''
        section_name = row[18] or ''
        
        # =====================================================================
        # QUERY 2: Fetch ALL actions ordered by DateAndTimeCreated ASC
        # =====================================================================
        cursor.execute("""
            SELECT 
                Description,
                Note,
                SectionNote,
                SelectionNote,
                DepartmentNote,
                ProblemReason,
                GoverningPolicies,
                DateAndTimeCreated
            FROM dbo.IncidentRequestCaseAction
            WHERE IncidentRequestCaseID = ?
            ORDER BY DateAndTimeCreated ASC
        """, legacy_case_id_val)
        
        action_rows = cursor.fetchall()
        
        # Parse actions into list of dicts
        actions = []
        for arow in action_rows:
            actions.append({
                "description": arow[0] or '',
                "note": arow[1] or '',
                "section_note": arow[2] or '',
                "selection_note": arow[3] or '',
                "department_note": arow[4] or '',
                "problem_reason": arow[5] or '',
                "governing_policies": arow[6] or '',
                "date_created": arow[7]
            })
        
        # =====================================================================
        # BUILD THE 3 TEXT FIELDS
        # =====================================================================
        
        # FIELD 1: complaint_content = CaseDescription + RequesterNote
        complaint_content = _build_complaint_content(case_description, requester_note)
        
        # FIELD 2: immediate_action = First action's fields
        first_action = actions[0] if actions else None
        immediate_action = _build_immediate_action(first_action)
        
        # FIELD 3: actions_taken = Remaining actions' fields
        remaining_actions = actions[1:] if len(actions) > 1 else []
        actions_taken = _build_actions_taken(remaining_actions)
        
        # =====================================================================
        # BUILD RESPONSE
        # =====================================================================
        org_name = department_name or section_name or 'N/A'
        
        return {
            # Core identifiers
            "legacy_case_id": legacy_case_id_val,
            
            # ============ THE 3 BUILT TEXT FIELDS ============
            "complaint_content": complaint_content,
            "complaint_text": complaint_content,  # Alias
            "immediate_action": immediate_action,
            "taken_action": actions_taken,
            "actions_taken": actions_taken,  # Alias
            
            # Preview object for frontend MigrationViewPage
            "preview": {
                "complaint_content": complaint_content,
                "complaint_text": complaint_content,
                "immediate_action": immediate_action,
                "taken_action": actions_taken,
                "actions_taken": actions_taken
            },
            
            # Basic metadata
            "patient_name": patient_name,
            "feedback_received_date": feedback_date,
            "feedback_date": feedback_date,
            "created_at": created_at,
            "is_inpatient": bool(is_inpatient) if is_inpatient is not None else None,
            "building": building,
            
            # Organization
            "department_id": department_id,
            "section_id": section_id,
            "issuing_org_unit_id": department_id or section_id,
            "issuing_department_id": department_id or section_id,
            "issuing_org_unit_name": org_name,
            "issuing_org_name": org_name,
            "department_name": org_name,
            
            # Legacy IDs (from old system - may not map to new lookup tables)
            "legacy_category_id": category_id,
            "legacy_subcategory_id": subcategory_id,
            "legacy_doctor_id": doctor_id,
            "case_status_id": status_id,
            
            # Migration status
            "migrated": migrated,
            "migrated_case_id": migrated_case_id,
            
            # Actions count for info
            "total_actions_count": len(actions),
            "remaining_actions_count": len(remaining_actions)
        }
        
    except Exception as e:
        raise Exception("Failed to get legacy case: " + str(e))
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
