"""
Database Layer: Ordinary Case Explanations
===========================================
Handles updating TakenAction field in APP_IncidentCase table.

For Ordinary cases (ClinicalRiskTypeID=1) with RequiresExplanation=1,
explanations are simpler - just append explanation text to TakenAction field.
"""

from typing import Dict, Any
from core.database import get_connection
from datetime import datetime


def update_ordinary_explanation(
    incident_request_case_id: int,
    explanation_text: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Update TakenAction field with explanation for ordinary cases.
    
    Args:
        incident_request_case_id: Case ID from APP_IncidentCase
        explanation_text: Explanation text to append to TakenAction
        user_id: ID of user submitting explanation
    
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
        
        # Validate case exists and is Ordinary
        cursor.execute(
            """
            SELECT 
                ClinicalRiskTypeID, 
                ExplanationStatusID, 
                CaseStatusID,
                RequiresExplanation,
                TakenAction
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
        requires_explanation = row.RequiresExplanation
        current_taken_action = row.TakenAction or ""
        
        # Validate it's Ordinary case
        if clinical_risk_type_id != 1:
            return {
                "success": False,
                "error": "INVALID_CASE_TYPE",
                "message": "This endpoint is only for Ordinary cases (ClinicalRiskTypeID=1)",
                "actual_type": clinical_risk_type_id,
                "hint": "Use /api/explanations/red-flag/{id} for Red Flag/Never Event cases"
            }
        
        # Validate it requires explanation
        if not requires_explanation:
            return {
                "success": False,
                "error": "EXPLANATION_NOT_REQUIRED",
                "message": "This case does not require explanation (RequiresExplanation=0)"
            }
        
        # Validate FSM state: Must be S0 (Open + Waiting)
        if not (case_status_id == 1 and (explanation_status_id == 1 or explanation_status_id is None)):
            return {
                "success": False,
                "error": "INVALID_STATE",
                "message": f"Case must be Open (1) and Waiting (1). Current: CaseStatus={case_status_id}, ExplanationStatus={explanation_status_id}"
            }
        
        # Append explanation to TakenAction with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "\n\n--- Explanation Added ---\n"
        new_taken_action = f"{current_taken_action}{separator}[{timestamp}] {explanation_text}"
        
        # Update case
        cursor.execute(
            """
            UPDATE dbo.APP_IncidentCase
            SET TakenAction = ?,
                CaseStatusID = 2,  -- In Progress
                ExplanationStatusID = 2  -- Responded
            WHERE IncidentRequestCaseID = ?
            """,
            (new_taken_action, incident_request_case_id)
        )
        
        conn.commit()
        
        return {
            "success": True,
            "message": "Ordinary case explanation submitted successfully",
            "updated_field": "TakenAction",
            "fsm_transition": "S0 → S1 (Open + Waiting → In Progress + Responded)"
        }
    
    except Exception as e:
        if conn:
            conn.rollback()
        return {
            "success": False,
            "error": "DATABASE_ERROR",
            "message": f"Failed to update explanation: {str(e)}"
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_ordinary_explanation(incident_request_case_id: int) -> Dict[str, Any]:
    """
    Retrieve TakenAction field for an ordinary case.
    
    Returns:
        Dictionary with case details or error
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT 
                IncidentRequestCaseID,
                TakenAction,
                ClinicalRiskTypeID,
                ExplanationStatusID,
                CaseStatusID,
                RequiresExplanation
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
        
        return {
            "success": True,
            "incident_request_case_id": row.IncidentRequestCaseID,
            "taken_action": row.TakenAction,
            "clinical_risk_type_id": row.ClinicalRiskTypeID,
            "explanation_status_id": row.ExplanationStatusID,
            "case_status_id": row.CaseStatusID,
            "requires_explanation": row.RequiresExplanation
        }
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
