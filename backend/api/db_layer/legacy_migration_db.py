"""
PHASE K — Legacy Migration DB Layer

Read-only database functions for accessing legacy case data.
These functions support the migration UI and mapping pipeline.

DO NOT modify legacy tables.
DO NOT perform migration writes here.
"""

from typing import Dict, List, Tuple, Any, Optional
from core.database import get_connection


def list_legacy_cases_paged(
    page: int,
    page_size: int
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Get paginated list of legacy cases eligible for migration.
    
    This is a READ-ONLY function used by the Migration Page to preview
    legacy data that has not yet been migrated.
    
    Args:
        page: Page number (1-indexed)
        page_size: Number of records per page
    
    Returns:
        Tuple of (rows, total_count):
            - rows: List of legacy case preview dicts
            - total_count: Total number of unmigrated legacy cases
    
    Row Structure:
        - legacy_case_id: int (IncidentRequestCase.UniqueID)
        - incident_request_id: int (IncidentRequest.UniqueID)
        - patient_name: str
        - received_date: str (ISO format)
        - preview_description: str (truncated to 200 chars)
        - source_section_id: int or None
        - source_department_id: int or None
        - source_admin_id: int or None
    
    Data Sources:
        - IncidentRequestCase (legacy case table)
        - IncidentRequest (legacy request table)
        - APP_DataMigration_Map (exclusion filter)
    
    Filter:
        Excludes cases already migrated (not in mapping table).
    
    Order:
        DateAndTimeRecieved DESC (newest first)
    
    Safety:
        - Read-only (no writes)
        - No schema changes
        - Connection safely closed
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Calculate offset for pagination
        offset = (page - 1) * page_size
        
        # Get total count of unmigrated cases
        count_query = """
            SELECT COUNT(*)
            FROM dbo.IncidentRequestCase irc
            INNER JOIN dbo.IncidentRequest ir 
                ON ir.UniqueID = irc.IncidentRequestID
            LEFT JOIN dbo.APP_DataMigration_Map map 
                ON map.legacy_case_id = irc.UniqueID
            WHERE map.legacy_case_id IS NULL
        """
        
        cursor.execute(count_query)
        total_count = cursor.fetchone()[0]
        
        # Get paginated results
        data_query = """
            SELECT 
                irc.UniqueID AS legacy_case_id,
                ir.UniqueID AS incident_request_id,
                ir.PatientName AS patient_name,
                CONVERT(VARCHAR(19), ir.DateAndTimeRecieved, 121) AS received_date,
                LEFT(irc.Description, 200) AS preview_description,
                ir.SourceSectionID AS source_section_id,
                ir.SourceDepartmentID AS source_department_id,
                ir.SourceAdminID AS source_admin_id
            FROM dbo.IncidentRequestCase irc
            INNER JOIN dbo.IncidentRequest ir 
                ON ir.UniqueID = irc.IncidentRequestID
            LEFT JOIN dbo.APP_DataMigration_Map map 
                ON map.legacy_case_id = irc.UniqueID
            WHERE map.legacy_case_id IS NULL
            ORDER BY ir.DateAndTimeRecieved DESC
            OFFSET ? ROWS
            FETCH NEXT ? ROWS ONLY
        """
        
        cursor.execute(data_query, (offset, page_size))
        
        rows = []
        for row in cursor.fetchall():
            rows.append({
                "legacy_case_id": row.legacy_case_id,
                "incident_request_id": row.incident_request_id,
                "patient_name": row.patient_name,
                "received_date": row.received_date,
                "preview_description": row.preview_description,
                "source_section_id": row.source_section_id,
                "source_department_id": row.source_department_id,
                "source_admin_id": row.source_admin_id
            })
        
        return (rows, total_count)
        
    except Exception as e:
        raise Exception(f"Failed to list legacy cases: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_legacy_case_detail(legacy_case_id: int) -> Optional[Dict[str, Any]]:
    """
    Get full detail of a single legacy case for migration preview.
    
    This is a READ-ONLY function that retrieves complete legacy case data
    including associated request information and all action history.
    
    Args:
        legacy_case_id: UniqueID from IncidentRequestCase table
    
    Returns:
        Dictionary with structure:
        {
            "case": {
                "UniqueID": int,
                "Description": str or None,
                "Note": str or None,
                "DoctorID": int or None,
                "SectionID": int or None,
                "DepartmentID": int or None,
                "AdminID": int or None,
                "DateAndTimeCreated": str or None (ISO format),
                "DateAndTimeUpdated": str or None (ISO format),
                "DateAndTimeHappened": str or None (ISO format),
                "IncidentTypeID": int or None
            },
            "request": {
                "PatientName": str or None,
                "MRN": str or None,
                "SourceBuilding": str or None,
                "IsInPatient": bool or None,
                "RequesterName": str or None,
                "Note": str or None,
                "DateAndTimeRecieved": str or None (ISO format),
                "SourceSectionID": int or None,
                "SourceDepartmentID": int or None,
                "SourceAdminID": int or None
            },
            "actions": [
                {
                    "UniqueID": int,
                    "Description": str or None,
                    "Note": str or None,
                    "SectionNote": str or None,
                    "DepartmentNote": str or None,
                    "SelectionNote": str or None,
                    "ProblemReason": str or None,
                    "GoverningPolicies": str or None,
                    "DateAndTimeCreated": str or None (ISO format)
                },
                ...
            ]
        }
        
        Returns None if case not found.
    
    Data Sources:
        - IncidentRequestCase (case data)
        - IncidentRequest (request data)
        - IncidentRequestCaseAction (action history)
    
    Actions are ordered by DateAndTimeCreated ASC, UniqueID ASC.
    
    Safety:
        - Read-only (no writes)
        - Parameterized queries
        - Safe connection handling
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query case + request data with JOIN
        case_query = """
            SELECT 
                irc.UniqueID,
                irc.Description,
                irc.Note,
                irc.DoctorID,
                irc.SectionID,
                irc.DepartmentID,
                irc.AdminID,
                CONVERT(VARCHAR(19), irc.DateAndTimeCreated, 121) AS DateAndTimeCreated,
                CONVERT(VARCHAR(19), irc.DateAndTimeUpdated, 121) AS DateAndTimeUpdated,
                CONVERT(VARCHAR(19), irc.DateAndTimeHappened, 121) AS DateAndTimeHappened,
                irc.IncidentTypeID,
                ir.PatientName,
                ir.MRN,
                ir.SourceBuilding,
                ir.IsInPatient,
                ir.RequesterName,
                ir.Note AS RequestNote,
                CONVERT(VARCHAR(19), ir.DateAndTimeRecieved, 121) AS DateAndTimeRecieved,
                ir.SourceSectionID,
                ir.SourceDepartmentID,
                ir.SourceAdminID
            FROM dbo.IncidentRequestCase irc
            INNER JOIN dbo.IncidentRequest ir 
                ON ir.UniqueID = irc.IncidentRequestID
            WHERE irc.UniqueID = ?
        """
        
        cursor.execute(case_query, (legacy_case_id,))
        case_row = cursor.fetchone()
        
        if not case_row:
            return None
        
        # Build case dict
        case_data = {
            "UniqueID": case_row[0],
            "Description": case_row[1],
            "Note": case_row[2],
            "DoctorID": case_row[3],
            "SectionID": case_row[4],
            "DepartmentID": case_row[5],
            "AdminID": case_row[6],
            "DateAndTimeCreated": case_row[7],
            "DateAndTimeUpdated": case_row[8],
            "DateAndTimeHappened": case_row[9],
            "IncidentTypeID": case_row[10]
        }
        
        # Build request dict
        request_data = {
            "PatientName": case_row[11],
            "MRN": case_row[12],
            "SourceBuilding": case_row[13],
            "IsInPatient": case_row[14],
            "RequesterName": case_row[15],
            "Note": case_row[16],
            "DateAndTimeRecieved": case_row[17],
            "SourceSectionID": case_row[18],
            "SourceDepartmentID": case_row[19],
            "SourceAdminID": case_row[20]
        }
        
        # Query actions for this case
        actions_query = """
            SELECT 
                UniqueID,
                Description,
                Note,
                SectionNote,
                DepartmentNote,
                SelectionNote,
                ProblemReason,
                GoverningPolicies,
                CONVERT(VARCHAR(19), DateAndTimeCreated, 121) AS DateAndTimeCreated
            FROM dbo.IncidentRequestCaseAction
            WHERE IncidentRequestCaseID = ?
            ORDER BY DateAndTimeCreated ASC, UniqueID ASC
        """
        
        cursor.execute(actions_query, (legacy_case_id,))
        
        actions = []
        for action_row in cursor.fetchall():
            actions.append({
                "UniqueID": action_row[0],
                "Description": action_row[1],
                "Note": action_row[2],
                "SectionNote": action_row[3],
                "DepartmentNote": action_row[4],
                "SelectionNote": action_row[5],
                "ProblemReason": action_row[6],
                "GoverningPolicies": action_row[7],
                "DateAndTimeCreated": action_row[8]
            })
        
        return {
            "case": case_data,
            "request": request_data,
            "actions": actions
        }
        
    except Exception as e:
        raise Exception(f"Failed to get legacy case detail: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
