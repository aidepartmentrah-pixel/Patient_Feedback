"""
DB Layer: Explanation Queries
==============================
Database query functions for the explanation workflow.
Handles read operations for cases requiring explanations.
"""

import pyodbc
from typing import Optional, List, Dict, Any
from datetime import datetime

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def _fetch_all(query: str, params: tuple = ()) -> List[Dict[str, Any]]:
    """Execute query and return all results as list of dictionaries"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    conn.close()
    return [dict(zip(columns, row)) for row in rows]


def _fetch_one(query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
    """Execute query and return single result as dictionary"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(query, params)
    row = cursor.fetchone()
    
    if not row:
        conn.close()
        return None
    
    columns = [col[0] for col in cursor.description]
    conn.close()
    return dict(zip(columns, row))


# -----------------------------
# LOOKUP QUERIES
# -----------------------------

def get_explanation_status_id(status_name: str) -> Optional[int]:
    """
    Get explanation status ID by name.
    
    Args:
        status_name: Name of status ('Waiting', 'Responded', 'Forcibly Closed', 'No Explanation Needed')
    
    Returns:
        Status ID or None if not found
    """
    result = _fetch_one(
        """
        SELECT StatusID
        FROM dbo.APP_LOOKUP_EXPLANATION_STATUS
        WHERE StatusName = ?
        """,
        (status_name,)
    )
    return result["StatusID"] if result else None


def get_explanation_status_name(status_id: int) -> Optional[str]:
    """
    Get explanation status name by ID.
    
    Args:
        status_id: Status ID
    
    Returns:
        Status name or None if not found
    """
    result = _fetch_one(
        """
        SELECT StatusName
        FROM dbo.APP_LOOKUP_EXPLANATION_STATUS
        WHERE StatusID = ?
        """,
        (status_id,)
    )
    return result["StatusName"] if result else None


def get_all_explanation_statuses() -> List[Dict[str, Any]]:
    """Get all explanation statuses"""
    return _fetch_all(
        """
        SELECT StatusID, StatusName
        FROM dbo.APP_LOOKUP_EXPLANATION_STATUS
        ORDER BY StatusID
        """
    )


def get_case_status_id(status_code: str) -> Optional[int]:
    """
    Get case status ID by code.
    
    Args:
        status_code: Status code ('OPEN', 'IN_PROGRESS', 'CLOSED')
    
    Returns:
        Status ID or None if not found
    """
    result = _fetch_one(
        """
        SELECT CaseStatusID
        FROM dbo.APP_LOOKUP_CASE_STATUS
        WHERE Code = ?
        """,
        (status_code,)
    )
    return result["CaseStatusID"] if result else None


def get_case_status_name(status_id: int) -> Optional[str]:
    """
    Get case status name by ID.
    
    Args:
        status_id: Status ID
    
    Returns:
        Status name or None if not found
    """
    result = _fetch_one(
        """
        SELECT Name
        FROM dbo.APP_LOOKUP_CASE_STATUS
        WHERE CaseStatusID = ?
        """,
        (status_id,)
    )
    return result["Name"] if result else None


# -----------------------------
# CASE QUERIES
# -----------------------------

def get_case_by_id(case_id: int) -> Optional[Dict[str, Any]]:
    """
    Get full incident case details by ID.
    
    Args:
        case_id: Incident case ID
    
    Returns:
        Dictionary with case details or None if not found
    """
    query = """
        SELECT 
            ic.IncidentRequestCaseID,
            ic.ComplaintText,
            ic.ImmediateAction,
            ic.TakenAction,
            ic.RequiresExplanation,
            ic.FeedbackRecievedDate,
            ic.PatientName,
            ic.IssuingOrgUnitID,
            ic.CreatedAt,
            ic.CreatedByUserID,
            ic.isINPatient,
            ic.ClinicalRiskTypeID,
            ic.FeedbackIntentTypeID,
            ic.BuildingID,
            ic.DomainID,
            ic.CategoryID,
            ic.SubCategoryID,
            ic.ClassificationID,
            ic.SeverityID,
            ic.StageID,
            ic.HarmLevelID,
            ic.CaseStatusID,
            ic.SourceID,
            ic.ExplanationStatusID,
            cs.Code AS CaseStatusCode,
            cs.Name AS CaseStatusName,
            es.StatusName AS ExplanationStatusName,
            crt.Name AS ClinicalRiskType,
            fit.NameEn AS FeedbackIntentType
        FROM dbo.APP_IncidentCase ic
        LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
        LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        LEFT JOIN dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE fit ON ic.FeedbackIntentTypeID = fit.FeedbackIntentTypeID
        WHERE ic.IncidentRequestCaseID = ?
    """
    return _fetch_one(query, (case_id,))


def get_cases_needing_explanation(
    department_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    case_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get all cases that need explanation but haven't received one yet.
    
    FIXED Criteria (matching FSM logic from insert_service.py):
    - CaseStatusID = 1  
    - (ClinicalRiskTypeID IN (2, 3) OR RequiresExplanation = 1)
    - (ExplanationStatusID IS NULL OR ExplanationStatusID = 1)  -- 1 = 'Waiting'
    
    This ensures Red Flag (2) and Never Event (3) cases appear even with NULL status.
    Only Open cases are returned because explanations can only be submitted for Open cases.
    
    Args:
        department_id: Filter by target department (optional)
        start_date: Filter by feedback received date >= start_date (optional)
        end_date: Filter by feedback received date <= end_date (optional)
        case_type: Filter by clinical risk type ('Red Flag', 'Never Event', etc.) (optional)
    
    Returns:
        List of cases needing explanation
    """
    query = """
        SELECT 
            ic.IncidentRequestCaseID,
            ic.ComplaintText,
            ic.ImmediateAction,
            ic.TakenAction,
            ic.RequiresExplanation,
            ic.FeedbackRecievedDate,
            ic.PatientName,
            ic.CreatedAt,
            ic.CaseStatusID,
            ic.ExplanationStatusID,
            ic.ClinicalRiskTypeID,
            cs.Code AS CaseStatusCode,
            cs.Name AS CaseStatusName,
            COALESCE(es.StatusName, 'Waiting') AS ExplanationStatusName,
            crt.Name AS ClinicalRiskType,
            fit.NameEn AS FeedbackIntentType,
            org.UniqueID AS IssuingOrgUnitID,
            org.Name AS IssuingOrgUnitName
        FROM dbo.APP_IncidentCase ic
        LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
        LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        LEFT JOIN dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE fit ON ic.FeedbackIntentTypeID = fit.FeedbackIntentTypeID
        LEFT JOIN dbo.AdminsrationUnit org ON ic.IssuingOrgUnitID = org.UniqueID
    """
    
    # Build WHERE clause - FIXED to match FSM logic
    # Only include Open cases (CaseStatusID = 1) that need explanations
    conditions = [
        "ic.CaseStatusID = 1", 
        "(ic.ClinicalRiskTypeID IN (2, 3) OR ic.RequiresExplanation = 1)",
        "(ic.ExplanationStatusID IS NULL OR ic.ExplanationStatusID = 1)"
    ]
    params = []
    
    if department_id is not None:
        # Join with target departments if filtering by department
        query += """
            INNER JOIN dbo.APP_IncidentCaseTargetDepartment ictd 
                ON ic.IncidentRequestCaseID = ictd.IncidentRequestCaseID
        """
        conditions.append("ictd.DepartmentID = ?")
        params.append(department_id)
    
    if start_date is not None:
        conditions.append("ic.FeedbackRecievedDate >= ?")
        params.append(start_date)
    
    if end_date is not None:
        conditions.append("ic.FeedbackRecievedDate <= ?")
        params.append(end_date)
    
    if case_type is not None:
        conditions.append("crt.Name = ?")
        params.append(case_type)
    
    query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY ic.FeedbackRecievedDate DESC, ic.CreatedAt DESC"
    
    return _fetch_all(query, tuple(params))


def get_red_flag_never_event_cases_needing_explanation(
    department_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[Dict[str, Any]]:
    """
    Get Red Flag and Never Event cases that are Open and need explanation.
    
    FIXED Criteria:
    - CaseStatusID = 1  -- Only Open cases
    - ClinicalRiskTypeID IN (2, 3)  -- 2 = Red Flag, 3 = Never Event
    - (ExplanationStatusID IS NULL OR ExplanationStatusID = 1)  -- NULL or Waiting
    
    Only Open cases are returned because Red Flag explanations can only be submitted for Open cases.
    
    Args:
        department_id: Filter by target department (optional)
        start_date: Filter by feedback received date >= start_date (optional)
        end_date: Filter by feedback received date <= end_date (optional)
    
    Returns:
        List of Red Flag/Never Event cases needing explanation
    """
    query = """
        SELECT 
            ic.IncidentRequestCaseID,
            ic.ComplaintText,
            ic.ImmediateAction,
            ic.TakenAction,
            ic.RequiresExplanation,
            ic.FeedbackRecievedDate,
            ic.PatientName,
            ic.CreatedAt,
            ic.CaseStatusID,
            ic.ExplanationStatusID,
            ic.ClinicalRiskTypeID,
            cs.Code AS CaseStatusCode,
            cs.Name AS CaseStatusName,
            COALESCE(es.StatusName, 'Waiting') AS ExplanationStatusName,
            crt.Name AS ClinicalRiskType,
            fit.NameEn AS FeedbackIntentType,
            org.UniqueID AS IssuingOrgUnitID,
            org.Name AS IssuingOrgUnitName
        FROM dbo.APP_IncidentCase ic
        LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
        LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        LEFT JOIN dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE fit ON ic.FeedbackIntentTypeID = fit.FeedbackIntentTypeID
        LEFT JOIN dbo.AdminsrationUnit org ON ic.IssuingOrgUnitID = org.UniqueID
    """
    
    # Build WHERE clause - FIXED to use numeric IDs
    # Only include Open cases (CaseStatusID = 1) that need explanations
    conditions = [
        "ic.CaseStatusID = 1", 
        "ic.ClinicalRiskTypeID IN (2, 3)",
        "(ic.ExplanationStatusID IS NULL OR ic.ExplanationStatusID = 1)"
    ]
    params = []
    
    if department_id is not None:
        query += """
            INNER JOIN dbo.APP_IncidentCaseTargetDepartment ictd 
                ON ic.IncidentRequestCaseID = ictd.IncidentRequestCaseID
        """
        conditions.append("ictd.DepartmentID = ?")
        params.append(department_id)
    
    if start_date is not None:
        conditions.append("ic.FeedbackRecievedDate >= ?")
        params.append(start_date)
    
    if end_date is not None:
        conditions.append("ic.FeedbackRecievedDate <= ?")
        params.append(end_date)
    
    query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY ic.FeedbackRecievedDate DESC, ic.CreatedAt DESC"
    
    return _fetch_all(query, tuple(params))


def count_cases_by_explanation_status() -> List[Dict[str, Any]]:
    """
    Count cases by explanation status.
    
    Returns:
        List of counts per status
    """
    return _fetch_all(
        """
        SELECT 
            es.StatusName,
            COUNT(ic.IncidentRequestCaseID) AS CaseCount
        FROM dbo.APP_LOOKUP_EXPLANATION_STATUS es
        LEFT JOIN dbo.APP_IncidentCase ic ON ic.ExplanationStatusID = es.StatusID
        GROUP BY es.StatusName
        ORDER BY es.StatusName
        """
    )


def get_overdue_explanations(days_threshold: int = 7) -> List[Dict[str, Any]]:
    """
    Get cases with overdue explanations.
    
    Overdue = RequiresExplanation = 1, ExplanationStatus = 'Waiting', 
              and FeedbackRecievedDate is older than threshold days
    
    Args:
        days_threshold: Number of days after which explanation is considered overdue
    
    Returns:
        List of overdue cases
    """
    return _fetch_all(
        """
        SELECT 
            ic.IncidentRequestCaseID,
            ic.ComplaintText,
            ic.FeedbackRecievedDate,
            ic.CreatedAt,
            DATEDIFF(day, ic.FeedbackRecievedDate, GETDATE()) AS DaysOverdue,
            crt.Name AS ClinicalRiskType,
            org.Name AS IssuingOrgUnitName
        FROM dbo.APP_IncidentCase ic
        LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
        LEFT JOIN dbo.APP_LOOKUP_CLINICAL_RISK_TYPE crt ON ic.ClinicalRiskTypeID = crt.ClinicalRiskTypeID
        LEFT JOIN dbo.AdminsrationUnit org ON ic.IssuingOrgUnitID = org.UniqueID
        WHERE ic.RequiresExplanation = 1
          AND es.StatusName = 'Waiting'
          AND DATEDIFF(day, ic.FeedbackRecievedDate, GETDATE()) >= ?
        ORDER BY DaysOverdue DESC
        """,
        (days_threshold,)
    )


def check_case_has_explanation(case_id: int) -> bool:
    """
    Check if a case has an explanation (TakenAction is not null/empty).
    
    Args:
        case_id: Incident case ID
    
    Returns:
        True if case has explanation, False otherwise
    """
    result = _fetch_one(
        """
        SELECT 
            CASE 
                WHEN TakenAction IS NOT NULL AND TakenAction != '' THEN 1 
                ELSE 0 
            END AS HasExplanation
        FROM dbo.APP_IncidentCase
        WHERE IncidentRequestCaseID = ?
        """,
        (case_id,)
    )
    return bool(result["HasExplanation"]) if result else False


# -----------------------------
# WRITE OPERATIONS
# -----------------------------

def update_case_explanation(
    case_id: int,
    explanation_text: str,
    user_id: int
) -> Dict[str, Any]:
    """
    Update case with explanation and transition FSM state.
    
    FSM Transition: Open + Waiting → In Progress + Responded
    
    Args:
        case_id: Incident case ID
        explanation_text: Explanation text to save in TakenAction field
        user_id: ID of user submitting explanation
    
    Returns:
        Dictionary with success status and updated case info
    
    Raises:
        ValueError: If case doesn't exist, is already closed, or doesn't require explanation
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Get current case state
        cursor.execute("""
            SELECT 
                ic.IncidentRequestCaseID,
                ic.RequiresExplanation,
                ic.CaseStatusID,
                ic.ExplanationStatusID,
                ic.ClinicalRiskTypeID,
                cs.Code AS CaseStatusCode,
                cs.IsFinal AS CaseStatusIsFinal,
                es.StatusName AS ExplanationStatusName
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
            WHERE ic.IncidentRequestCaseID = ?
        """, (case_id,))
        
        case = cursor.fetchone()
        
        if not case:
            conn.close()
            raise ValueError(f"Case {case_id} not found")
        
        # 2. Validate case state
        # Red Flag (2) or Never Event (3) ALWAYS need explanation, regardless of RequiresExplanation flag
        is_red_flag_or_never_event = case[4] in (2, 3)  # ClinicalRiskTypeID
        requires_explanation = is_red_flag_or_never_event or case[1]  # RequiresExplanation
        
        if not requires_explanation:
            conn.close()
            raise ValueError(f"Case {case_id} does not require explanation")
        
        if case[6]:  # CaseStatusIsFinal (is closed)
            conn.close()
            raise ValueError(f"Case {case_id} is already closed and cannot be updated")
        
        # 3. Get status IDs for transition
        in_progress_id = get_case_status_id("IN_PROGRESS")
        responded_id = get_explanation_status_id("Responded")
        
        if not in_progress_id or not responded_id:
            conn.close()
            raise ValueError("Could not find required status IDs for FSM transition")
        
        # 4. Update case with explanation and transition state
        cursor.execute("""
            UPDATE dbo.APP_IncidentCase
            SET 
                TakenAction = ?,
                CaseStatusID = ?,
                ExplanationStatusID = ?
            WHERE IncidentRequestCaseID = ?
        """, (explanation_text, in_progress_id, responded_id, case_id))
        
        conn.commit()
        
        # 5. Fetch updated case
        updated_case = get_case_by_id(case_id)
        
        conn.close()
        
        return {
            "success": True,
            "case_id": case_id,
            "message": "Explanation submitted successfully",
            "case": updated_case
        }
        
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        raise


def update_case_requires_explanation(
    case_id: int,
    requires_explanation: bool,
    user_id: int
) -> Dict[str, Any]:
    """
    Update the RequiresExplanation flag for a case.
    
    Args:
        case_id: Incident case ID
        requires_explanation: True to require explanation, False otherwise
        user_id: ID of user making the change
    
    Returns:
        Dictionary with success status
    
    Raises:
        ValueError: If case doesn't exist or is already closed
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Get current case state
        cursor.execute("""
            SELECT 
                ic.IncidentRequestCaseID,
                ic.RequiresExplanation,
                cs.IsFinal AS CaseStatusIsFinal
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE ic.IncidentRequestCaseID = ?
        """, (case_id,))
        
        case = cursor.fetchone()
        
        if not case:
            conn.close()
            raise ValueError(f"Case {case_id} not found")
        
        # 2. Validate: Cannot change if already closed
        if case[2]:  # CaseStatusIsFinal
            conn.close()
            raise ValueError(f"Cannot change RequiresExplanation for closed case {case_id}")
        
        # 3. Update flag
        cursor.execute("""
            UPDATE dbo.APP_IncidentCase
            SET RequiresExplanation = ?
            WHERE IncidentRequestCaseID = ?
        """, (1 if requires_explanation else 0, case_id))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "case_id": case_id,
            "requires_explanation": requires_explanation,
            "message": f"RequiresExplanation set to {requires_explanation}"
        }
        
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        raise


def force_close_case(
    case_id: int,
    user_id: int,
    reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Force close a case (admin override).
    
    FSM Transition: Any state → Closed + Forcibly Closed
    
    Args:
        case_id: Incident case ID
        user_id: ID of admin user forcing closure
        reason: Optional reason for force closure
    
    Returns:
        Dictionary with success status
    
    Raises:
        ValueError: If case doesn't exist or is already closed
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Get current case state
        cursor.execute("""
            SELECT 
                ic.IncidentRequestCaseID,
                cs.IsFinal AS CaseStatusIsFinal
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            WHERE ic.IncidentRequestCaseID = ?
        """, (case_id,))
        
        case = cursor.fetchone()
        
        if not case:
            conn.close()
            raise ValueError(f"Case {case_id} not found")
        
        # 2. Check if already closed
        if case[1]:  # CaseStatusIsFinal
            conn.close()
            raise ValueError(f"Case {case_id} is already closed")
        
        # 3. Get status IDs for force close
        closed_id = get_case_status_id("CLOSED")
        forcibly_closed_id = get_explanation_status_id("Forcibly Closed")
        
        if not closed_id or not forcibly_closed_id:
            conn.close()
            raise ValueError("Could not find required status IDs for force closure")
        
        # 4. Update case to force closed state
        cursor.execute("""
            UPDATE dbo.APP_IncidentCase
            SET 
                CaseStatusID = ?,
                ExplanationStatusID = ?
            WHERE IncidentRequestCaseID = ?
        """, (closed_id, forcibly_closed_id, case_id))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "case_id": case_id,
            "message": "Case forcibly closed by admin",
            "reason": reason
        }
        
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        raise


def close_case_after_action_items(
    case_id: int,
    user_id: int
) -> Dict[str, Any]:
    """
    Close a case after all action items are completed.
    
    FSM Transition: In Progress + Responded → Closed + Responded
    
    Args:
        case_id: Incident case ID
        user_id: ID of user closing the case
    
    Returns:
        Dictionary with success status
    
    Raises:
        ValueError: If case doesn't exist, is not in correct state, or has incomplete action items
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # 1. Get current case state
        cursor.execute("""
            SELECT 
                ic.IncidentRequestCaseID,
                ic.CaseStatusID,
                ic.ExplanationStatusID,
                cs.Code AS CaseStatusCode,
                es.StatusName AS ExplanationStatusName
            FROM dbo.APP_IncidentCase ic
            LEFT JOIN dbo.APP_LOOKUP_CASE_STATUS cs ON ic.CaseStatusID = cs.CaseStatusID
            LEFT JOIN dbo.APP_LOOKUP_EXPLANATION_STATUS es ON ic.ExplanationStatusID = es.StatusID
            WHERE ic.IncidentRequestCaseID = ?
        """, (case_id,))
        
        case = cursor.fetchone()
        
        if not case:
            conn.close()
            raise ValueError(f"Case {case_id} not found")
        
        # 2. Validate state: Must be In Progress + Responded
        if case[3] != "IN_PROGRESS":  # CaseStatusCode
            conn.close()
            raise ValueError(f"Case {case_id} must be 'In Progress' to close (current: {case[3]})")
        
        if case[4] != "Responded":  # ExplanationStatusName
            conn.close()
            raise ValueError(f"Case {case_id} must have 'Responded' explanation status (current: {case[4]})")
        
        # 3. Check if all action items are completed
        cursor.execute("""
            SELECT COUNT(*) AS IncompleteCount
            FROM dbo.APP_ActionItem
            WHERE IncidentRequestCaseID = ?
              AND IsDone = 0
        """, (case_id,))
        
        incomplete_count = cursor.fetchone()[0]
        
        if incomplete_count > 0:
            conn.close()
            raise ValueError(f"Case {case_id} has {incomplete_count} incomplete action item(s)")
        
        # 4. Get status IDs for closure
        closed_id = get_case_status_id("CLOSED")
        responded_id = get_explanation_status_id("Responded")
        
        if not closed_id:
            conn.close()
            raise ValueError("Could not find CLOSED status ID")
        
        # 5. Close the case
        cursor.execute("""
            UPDATE dbo.APP_IncidentCase
            SET 
                CaseStatusID = ?,
                ExplanationStatusID = ?
            WHERE IncidentRequestCaseID = ?
        """, (closed_id, responded_id, case_id))
        
        conn.commit()
        conn.close()
        
        return {
            "success": True,
            "case_id": case_id,
            "message": "Case closed successfully after action items completion"
        }
        
    except Exception as e:
        try:
            conn.rollback()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        raise

