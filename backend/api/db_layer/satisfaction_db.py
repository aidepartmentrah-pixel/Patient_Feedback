"""
Satisfaction Database Layer
Handles SQL operations for APP_IncidentCaseSatisfaction and APP_Lookup_SatisfactionStatus tables.

NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from core.database import get_connection


# ============================================================
# LOOKUP TABLE
# ============================================================

def get_satisfaction_statuses() -> List[Dict[str, Any]]:
    """
    Get all satisfaction status lookup values.
    
    Returns:
        List of status dictionaries with id, name_en, name_ar
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                SatisfactionStatusID,
                StatusNameEn,
                StatusNameAr,
                IsActive
            FROM dbo.APP_Lookup_SatisfactionStatus
            WHERE IsActive = 1
            ORDER BY SatisfactionStatusID
        """)
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'satisfaction_status_id': row.SatisfactionStatusID,
                'status_name_en': row.StatusNameEn,
                'status_name_ar': row.StatusNameAr,
            })
        
        return results
    finally:
        cursor.close()
        conn.close()


# ============================================================
# CREATE SATISFACTION
# ============================================================

def _has_feedback_text_column() -> bool:
    """Check if FeedbackText column exists in the table."""
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_IncidentCaseSatisfaction' AND COLUMN_NAME = 'FeedbackText'
        """)
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        conn.close()


def create_satisfaction(
    incident_case_id: int,
    feedback_needed: bool,
    feedback_given: bool,
    feedback_datetime: Optional[datetime],
    satisfaction_status_id: int,
    created_by_user_id: int,
    feedback_text: Optional[str] = None
) -> int:
    """
    Create a satisfaction record for an incident case.
    
    Args:
        incident_case_id: The incident case ID
        feedback_needed: Whether feedback was requested
        feedback_given: Whether feedback was given
        feedback_datetime: When feedback was given (if applicable)
        satisfaction_status_id: Status from lookup table (1=Not Present, 2=Satisfied, 3=Not Satisfied)
        created_by_user_id: User creating the record
        feedback_text: Optional feedback text/notes
        
    Returns:
        SatisfactionID of created record
        
    Raises:
        Exception: If satisfaction already exists for this case (unique constraint)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if FeedbackText column exists
    has_feedback_text_col = _has_feedback_text_column()
    
    try:
        if has_feedback_text_col and feedback_text:
            cursor.execute("""
                INSERT INTO dbo.APP_IncidentCaseSatisfaction (
                    IncidentRequestCaseID,
                    FeedbackNeeded,
                    FeedbackGiven,
                    FeedbackDateTime,
                    FeedbackText,
                    SatisfactionStatusID,
                    CreatedByUserID,
                    CreatedAt
                )
                OUTPUT INSERTED.SatisfactionID
                VALUES (?, ?, ?, ?, ?, ?, ?, GETDATE())
            """, (
                incident_case_id,
                feedback_needed,
                feedback_given,
                feedback_datetime,
                feedback_text,
                satisfaction_status_id,
                created_by_user_id
            ))
        else:
            cursor.execute("""
                INSERT INTO dbo.APP_IncidentCaseSatisfaction (
                    IncidentRequestCaseID,
                    FeedbackNeeded,
                    FeedbackGiven,
                    FeedbackDateTime,
                    SatisfactionStatusID,
                    CreatedByUserID,
                    CreatedAt
                )
                OUTPUT INSERTED.SatisfactionID
                VALUES (?, ?, ?, ?, ?, ?, GETDATE())
            """, (
                incident_case_id,
                feedback_needed,
                feedback_given,
                feedback_datetime,
                satisfaction_status_id,
                created_by_user_id
            ))
        
        result = cursor.fetchone()
        conn.commit()
        
        return result.SatisfactionID if result else None
    except Exception as e:
        conn.rollback()
        if 'UQ_Satisfaction_Case' in str(e) or 'UNIQUE KEY' in str(e).upper():
            raise Exception(f"Satisfaction already exists for case {incident_case_id}")
        raise
    finally:
        cursor.close()
        conn.close()


# ============================================================
# READ SATISFACTION
# ============================================================

def get_satisfaction_by_case(incident_case_id: int) -> Optional[Dict[str, Any]]:
    """
    Get satisfaction record for a specific incident case.
    
    Args:
        incident_case_id: The incident case ID
        
    Returns:
        Satisfaction dict or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 
                s.SatisfactionID,
                s.IncidentRequestCaseID,
                s.FeedbackNeeded,
                s.FeedbackGiven,
                s.FeedbackDateTime,
                s.SatisfactionStatusID,
                ls.StatusNameEn,
                ls.StatusNameAr,
                s.CreatedByUserID,
                s.CreatedAt
            FROM dbo.APP_IncidentCaseSatisfaction s
            INNER JOIN dbo.APP_Lookup_SatisfactionStatus ls
                ON s.SatisfactionStatusID = ls.SatisfactionStatusID
            WHERE s.IncidentRequestCaseID = ?
        """, incident_case_id)
        
        row = cursor.fetchone()
        
        if not row:
            return None
            
        return {
            'satisfaction_id': row.SatisfactionID,
            'incident_case_id': row.IncidentRequestCaseID,
            'feedback_needed': row.FeedbackNeeded,
            'feedback_given': row.FeedbackGiven,
            'feedback_datetime': row.FeedbackDateTime.isoformat() if row.FeedbackDateTime else None,
            'satisfaction_status_id': row.SatisfactionStatusID,
            'satisfaction_status_name_en': row.StatusNameEn,
            'satisfaction_status_name_ar': row.StatusNameAr,
            'created_by_user_id': row.CreatedByUserID,
            'created_at': row.CreatedAt.isoformat() if row.CreatedAt else None,
        }
    finally:
        cursor.close()
        conn.close()


def get_satisfactions_by_cases(incident_case_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Get satisfaction records for multiple incident cases.
    Used for batch loading in patient history.
    
    Args:
        incident_case_ids: List of incident case IDs
        
    Returns:
        Dictionary mapping case_id -> satisfaction dict
    """
    if not incident_case_ids:
        return {}
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        placeholders = ','.join('?' * len(incident_case_ids))
        cursor.execute(f"""
            SELECT 
                s.SatisfactionID,
                s.IncidentRequestCaseID,
                s.FeedbackNeeded,
                s.FeedbackGiven,
                s.FeedbackDateTime,
                s.SatisfactionStatusID,
                ls.StatusNameEn,
                ls.StatusNameAr,
                s.CreatedByUserID,
                s.CreatedAt
            FROM dbo.APP_IncidentCaseSatisfaction s
            INNER JOIN dbo.APP_Lookup_SatisfactionStatus ls
                ON s.SatisfactionStatusID = ls.SatisfactionStatusID
            WHERE s.IncidentRequestCaseID IN ({placeholders})
        """, tuple(incident_case_ids))
        
        results = {}
        for row in cursor.fetchall():
            results[row.IncidentRequestCaseID] = {
                'satisfaction_id': row.SatisfactionID,
                'incident_case_id': row.IncidentRequestCaseID,
                'feedback_needed': row.FeedbackNeeded,
                'feedback_given': row.FeedbackGiven,
                'feedback_datetime': row.FeedbackDateTime.isoformat() if row.FeedbackDateTime else None,
                'satisfaction_status_id': row.SatisfactionStatusID,
                'satisfaction_status_name_en': row.StatusNameEn,
                'satisfaction_status_name_ar': row.StatusNameAr,
                'created_by_user_id': row.CreatedByUserID,
                'created_at': row.CreatedAt.isoformat() if row.CreatedAt else None,
            }
        
        return results
    finally:
        cursor.close()
        conn.close()


def satisfaction_exists(incident_case_id: int) -> bool:
    """
    Check if satisfaction record exists for a case.
    
    Args:
        incident_case_id: The incident case ID
        
    Returns:
        True if exists, False otherwise
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT 1 FROM dbo.APP_IncidentCaseSatisfaction
            WHERE IncidentRequestCaseID = ?
        """, incident_case_id)
        
        return cursor.fetchone() is not None
    finally:
        cursor.close()
        conn.close()


# ============================================================
# UPDATE SATISFACTION
# ============================================================

def update_satisfaction(
    incident_case_id: int,
    feedback_needed: bool,
    feedback_given: bool,
    feedback_datetime: Optional[datetime],
    satisfaction_status_id: int,
    modified_by_user_id: int,
    feedback_text: Optional[str] = None
) -> bool:
    """
    Update an existing satisfaction record for an incident case.
    
    Args:
        incident_case_id: The incident case ID
        feedback_needed: Whether feedback was requested
        feedback_given: Whether feedback was given
        feedback_datetime: When feedback was given (if applicable)
        satisfaction_status_id: Status from lookup table (1=Not Present, 2=Satisfied, 3=Not Satisfied)
        modified_by_user_id: User updating the record
        feedback_text: Optional feedback text/notes
        
    Returns:
        True if updated successfully, False if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if FeedbackText column exists
    has_feedback_text_col = _has_feedback_text_column()
    
    try:
        if has_feedback_text_col:
            cursor.execute("""
                UPDATE dbo.APP_IncidentCaseSatisfaction
                SET FeedbackNeeded = ?,
                    FeedbackGiven = ?,
                    FeedbackDateTime = ?,
                    FeedbackText = ?,
                    SatisfactionStatusID = ?
                WHERE IncidentRequestCaseID = ?
            """, (
                feedback_needed,
                feedback_given,
                feedback_datetime,
                feedback_text,
                satisfaction_status_id,
                incident_case_id
            ))
        else:
            cursor.execute("""
                UPDATE dbo.APP_IncidentCaseSatisfaction
                SET FeedbackNeeded = ?,
                    FeedbackGiven = ?,
                    FeedbackDateTime = ?,
                    SatisfactionStatusID = ?
                WHERE IncidentRequestCaseID = ?
            """, (
                feedback_needed,
                feedback_given,
                feedback_datetime,
                satisfaction_status_id,
                incident_case_id
            ))
        
        updated = cursor.rowcount > 0
        conn.commit()
        
        return updated
    except Exception as e:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()
