"""
DB Layer: Incident Case Employee
Handles employee linkage to incidents (Many-to-Many)

Schema: APP_IncidentCaseEmployee
  PK: ID (auto-increment IDENTITY)
  UNIQUE: (EmployeeID, IncidentRequestCaseID)
  FK: IncidentRequestCaseID -> APP_IncidentCase

An employee can be linked to MULTIPLE incidents.
An incident can have MULTIPLE employees.
"""

from backend.core.database import get_connection
from datetime import datetime


def add_employee_to_case(
    incident_id: int,
    employee_id: int,
    assigned_by_user_id: int,
    full_name: str = "",
    is_primary: bool = False
) -> int:
    """
    Link an employee to an incident case (many-to-many).
    
    Creates a new linkage row. If the same employee is already linked
    to the same incident, the existing record is updated instead.
    
    Args:
        incident_id: The incident case ID
        employee_id: The employee ID
        assigned_by_user_id: User who assigned this employee
        full_name: Employee full name (optional)
        is_primary: Whether this is the primary employee
    
    Returns:
        The new linkage ID (auto-increment)
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Check if this specific (employee, incident) pair already exists
        cursor.execute(
            """SELECT ID FROM dbo.APP_IncidentCaseEmployee 
               WHERE EmployeeID = ? AND IncidentRequestCaseID = ?""",
            (employee_id, incident_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing linkage (same employee + same incident)
            cursor.execute(
                """
                UPDATE dbo.APP_IncidentCaseEmployee
                SET 
                    IsPrimary = ?,
                    AssignedAt = GETDATE(),
                    AssignedByUserID = ?,
                    FullName = ?
                WHERE EmployeeID = ? AND IncidentRequestCaseID = ?
                """,
                (1 if is_primary else 0, assigned_by_user_id, full_name,
                 employee_id, incident_id)
            )
            conn.commit()
            return existing.ID
        else:
            # Insert NEW linkage row (employee can now link to multiple incidents)
            cursor.execute(
                """
                INSERT INTO dbo.APP_IncidentCaseEmployee (
                    EmployeeID,
                    FullName,
                    IncidentRequestCaseID,
                    IsPrimary,
                    AssignedAt,
                    AssignedByUserID
                )
                OUTPUT INSERTED.ID
                VALUES (?, ?, ?, ?, GETDATE(), ?)
                """,
                (employee_id, full_name, incident_id, 1 if is_primary else 0, assigned_by_user_id)
            )
            row = cursor.fetchone()
            new_id = row.ID if row else 0
            conn.commit()
            return new_id
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to add employee to case: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_employees_for_incident(incident_id: int) -> list:
    """
    Get all employees linked to an incident
    
    Args:
        incident_id: The incident case ID
    
    Returns:
        List of employee dictionaries
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT 
                EmployeeID,
                FullName,
                JobTitle,
                IsPrimary,
                AssignedAt
            FROM dbo.APP_IncidentCaseEmployee
            WHERE IncidentRequestCaseID = ?
            ORDER BY IsPrimary DESC, AssignedAt
            """,
            (incident_id,)
        )
        
        employees = []
        for row in cursor.fetchall():
            employees.append({
                "employee_id": row.EmployeeID,
                "full_name": row.FullName,
                "job_title": row.JobTitle if hasattr(row, 'JobTitle') else None,
                "is_primary": bool(row.IsPrimary),
                "assigned_at": row.AssignedAt.isoformat() if row.AssignedAt else None
            })
        
        return employees
        
    except Exception as e:
        raise Exception(f"Failed to get employees for incident: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_incidents_for_employee(employee_id: int) -> list:
    """
    Get all incidents linked to an employee
    
    Args:
        employee_id: The employee ID
    
    Returns:
        List of incident IDs
    """
    conn = None
    cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            """
            SELECT DISTINCT IncidentRequestCaseID
            FROM dbo.APP_IncidentCaseEmployee
            WHERE EmployeeID = ?
                AND IncidentRequestCaseID IS NOT NULL
            ORDER BY IncidentRequestCaseID
            """,
            (employee_id,)
        )
        
        incident_ids = [row.IncidentRequestCaseID for row in cursor.fetchall()]
        return incident_ids
        
    except Exception as e:
        raise Exception(f"Failed to get incidents for employee: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
