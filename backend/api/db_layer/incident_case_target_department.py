from core.database import get_connection

# -----------------------------
# CREATE
# -----------------------------

def add_target_department(
    incident_id: int,
    department_id: int,
    assigned_by_user_id: int,
    is_primary: bool = False,
) -> int:
    """
    Assign a target department to an incident case.
    If is_primary=True, all other target departments for the case are unset.
    Returns TargetID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if is_primary:
        cursor.execute(
            """
            UPDATE dbo.APP_IncidentCaseTargetDepartment
            SET IsPrimary = 0
            WHERE IncidentRequestCaseID = ?
            """,
            incident_id,
        )

    cursor.execute(
        """
        INSERT INTO dbo.APP_IncidentCaseTargetDepartment (
            IncidentRequestCaseID,
            DepartmentID,
            IsPrimary,
            AssignedByUserID
        )
        OUTPUT INSERTED.TargetID
        VALUES (?, ?, ?, ?)
        """,
        incident_id,
        department_id,
        int(is_primary),
        assigned_by_user_id,
    )

    target_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()

    return target_id


# -----------------------------
# READ
# -----------------------------

def list_target_departments(incident_id: int) -> list[dict]:
    """
    List all target departments assigned to a case.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_IncidentCaseTargetDepartment
        WHERE IncidentRequestCaseID = ?
        ORDER BY IsPrimary DESC, AssignedAt ASC
        """,
        incident_id,
    )

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()

    return [dict(zip(columns, row)) for row in rows]


# -----------------------------
# UPDATE
# -----------------------------

def set_primary_department(
    incident_id: int,
    target_id: int,
) -> None:
    """
    Set a specific target department as primary.
    Ensures only one primary department per case.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Unset all
    cursor.execute(
        """
        UPDATE dbo.APP_IncidentCaseTargetDepartment
        SET IsPrimary = 0
        WHERE IncidentRequestCaseID = ?
        """,
        incident_id,
    )

    # Set selected
    cursor.execute(
        """
        UPDATE dbo.APP_IncidentCaseTargetDepartment
        SET IsPrimary = 1
        WHERE TargetID = ?
          AND IncidentRequestCaseID = ?
        """,
        target_id,
        incident_id,
    )

    conn.commit()
    conn.close()


# -----------------------------
# DELETE
# -----------------------------

def remove_target_department(
    target_id: int,
) -> None:
    """
    Remove a target department assignment from a case.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM dbo.APP_IncidentCaseTargetDepartment
        WHERE TargetID = ?
        """,
        target_id,
    )

    conn.commit()
    conn.close()
