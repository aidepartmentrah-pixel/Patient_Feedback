"""
Database Layer: Red Flag/Never Event Explanations
==================================================
Handles creating feedback records in APP_IncidentCaseFeedback table.

For Red Flag (ClinicalRiskTypeID=2) and Never Event (ClinicalRiskTypeID=3) cases,
explanations require creating a new structured feedback record with:
- Root cause analysis (Staff, Process, Equipment, Environment)
- Preventive action plans
- Department explanation text
"""

from typing import Dict, Any, Optional
from core.database import get_connection
from datetime import datetime


def create_red_flag_feedback(
    incident_request_case_id: int,
    explanation_text: str,
    causes: Dict[str, Any],
    preventive_actions: Dict[str, Any],
    user_id: int
) -> Dict[str, Any]:
    """
    Create comprehensive feedback record for Red Flag/Never Event cases.
    
    Args:
        incident_request_case_id: Case ID from APP_IncidentCase
        explanation_text: Detailed explanation text
        causes: Dictionary with keys:
            - staff: {training, incentives, competency, understaffed, non_compliance, no_coordination, other, other_text}
            - process: {not_comprehensive, unclear, missing_protocol, other, other_text}
            - equipment: {not_available, system_incomplete, hard_to_apply, other, other_text}
            - environment: {place_nature, surroundings, work_conditions, other, other_text}
        preventive_actions: Dictionary with keys:
            - monthly_meetings, training_programs, increase_staff, mm_committee_actions, other, other_text
        user_id: ID of user submitting feedback
    
    Returns:
        Success/error dictionary
    
    FSM Transition:
        S0 (Open + Waiting) → S1 (In Progress + Responded)
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Validate case exists and is Red Flag/Never Event
        cursor.execute(
            """
            SELECT ClinicalRiskTypeID, ExplanationStatusID, CaseStatusID
            FROM dbo.APP_IncidentCase
            WHERE IncidentRequestCaseID = ?
            """,
            (incident_request_case_id,)
        )
        row = cursor.fetchone()
        
        if not row:
            return {
                "success": False,
                "error": "CASE_NOT_FOUND",
                "message": "Case not found"
            }
        
        clinical_risk_type_id = row.ClinicalRiskTypeID
        explanation_status_id = row.ExplanationStatusID
        case_status_id = row.CaseStatusID
        
        # Validate it's Red Flag or Never Event
        if clinical_risk_type_id not in (2, 3):
            return {
                "success": False,
                "error": "INVALID_CASE_TYPE",
                "message": "This endpoint is only for Red Flag (2) or Never Event (3) cases",
                "actual_type": clinical_risk_type_id
            }
        
        # Validate FSM state: Must be S0 (Open + Waiting)
        if not (case_status_id == 1 and (explanation_status_id == 1 or explanation_status_id is None)):
            return {
                "success": False,
                "error": "INVALID_STATE",
                "message": f"Case must be Open (1) and Waiting (1). Current: CaseStatus={case_status_id}, ExplanationStatus={explanation_status_id}"
            }
        
        # Check if feedback already exists
        cursor.execute(
            "SELECT COUNT(*) FROM dbo.APP_IncidentCaseFeedback WHERE IncidentRequestCaseID = ?",
            (incident_request_case_id,)
        )
        if cursor.fetchone()[0] > 0:
            return {
                "success": False,
                "error": "FEEDBACK_EXISTS",
                "message": "Feedback already submitted for this case"
            }
        
        # Extract cause data
        staff = causes.get('staff', {})
        process = causes.get('process', {})
        equipment = causes.get('equipment', {})
        environment = causes.get('environment', {})
        
        # Insert feedback record
        cursor.execute(
            """
            INSERT INTO dbo.APP_IncidentCaseFeedback (
                IncidentRequestCaseID,
                
                Cause_Staff_Training,
                Cause_Staff_Incentives,
                Cause_Staff_Competency,
                Cause_Staff_Understaffed,
                Cause_Staff_NonCompliance,
                Cause_Staff_NoCoordination,
                Cause_Staff_Other,
                Cause_Staff_OtherText,
                
                Cause_Process_NotComprehensive,
                Cause_Process_Unclear,
                Cause_Process_MissingProtocol,
                Cause_Process_Other,
                Cause_Process_OtherText,
                
                Cause_Equipment_NotAvailable,
                Cause_Equipment_SystemIncomplete,
                Cause_Equipment_HardToApply,
                Cause_Equipment_Other,
                Cause_Equipment_OtherText,
                
                Cause_Environment_PlaceNature,
                Cause_Environment_Surroundings,
                Cause_Environment_WorkConditions,
                Cause_Environment_Other,
                Cause_Environment_OtherText,
                
                Preventive_MonthlyMeetings,
                Preventive_TrainingPrograms,
                Preventive_IncreaseStaff,
                Preventive_MMCommitteeActions,
                Preventive_Other,
                Preventive_OtherText,
                
                DepartmentExplanationText,
                DepartmentExplanationStatusID,
                DepartmentExplanationReceivalDate,
                
                CreatedByUserID
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, ?, ?,
                   ?, ?, ?, ?, ?,
                   ?, ?, ?, ?, ?,
                   ?, ?, ?, ?, ?, ?,
                   ?, ?, ?,
                ?
            )
            """,
            (
                incident_request_case_id,
                
                staff.get('training', 0),
                staff.get('incentives', 0),
                staff.get('competency', 0),
                staff.get('understaffed', 0),
                staff.get('non_compliance', 0),
                staff.get('no_coordination', 0),
                staff.get('other', 0),
                staff.get('other_text'),
                
                process.get('not_comprehensive', 0),
                process.get('unclear', 0),
                process.get('missing_protocol', 0),
                process.get('other', 0),
                process.get('other_text'),
                
                equipment.get('not_available', 0),
                equipment.get('system_incomplete', 0),
                equipment.get('hard_to_apply', 0),
                equipment.get('other', 0),
                equipment.get('other_text'),
                
                environment.get('place_nature', 0),
                environment.get('surroundings', 0),
                environment.get('work_conditions', 0),
                environment.get('other', 0),
                environment.get('other_text'),
                
                preventive_actions.get('monthly_meetings', 0),
                preventive_actions.get('training_programs', 0),
                preventive_actions.get('increase_staff', 0),
                preventive_actions.get('mm_committee_actions', 0),
                preventive_actions.get('other', 0),
                preventive_actions.get('other_text'),
                
                explanation_text,
                2,  # DepartmentExplanationStatusID = 2 (Responded)
                datetime.now().date(),
                
                user_id
            )
        )
        
        # FSM Transition: S0 → S1
        cursor.execute(
            """
            UPDATE dbo.APP_IncidentCase
            SET CaseStatusID = 2,  -- In Progress
                ExplanationStatusID = 2  -- Responded
            WHERE IncidentRequestCaseID = ?
            """,
            (incident_request_case_id,)
        )
        
        conn.commit()
        
        return {
            "success": True,
            "message": "Red Flag/Never Event feedback submitted successfully",
            "feedback_created": True,
            "fsm_transition": "S0 → S1 (Open + Waiting → In Progress + Responded)"
        }
    
    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": "DATABASE_ERROR",
            "message": f"Failed to create feedback: {str(e)}"
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_red_flag_feedback_details(incident_request_case_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve existing feedback record for a Red Flag/Never Event case.
    
    Returns:
        Dictionary with feedback details or None if not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT *
            FROM dbo.APP_IncidentCaseFeedback
            WHERE IncidentRequestCaseID = ?
            """,
            (incident_request_case_id,)
        )
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        columns = [col[0] for col in cursor.description]
        return dict(zip(columns, row))
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
