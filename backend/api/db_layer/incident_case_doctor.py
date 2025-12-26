import pyodbc

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
# CREATE
# -----------------------------

def add_doctor_to_case(
    incident_id: int,
    doctor_id: int,
    assigned_by_user_id: int,
    is_primary: bool = False,
) -> int:
    """
    Assign a doctor to an incident case.
    If is_primary=True, all other doctors for the case are unset as primary.
    Returns IncidentCaseDoctorID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if is_primary:
        cursor.execute(
            """
            UPDATE dbo.APP_IncidentCaseDoctor
            SET IsPrimary = 0
            WHERE IncidentRequestCaseID = ?
            """,
            incident_id,
        )

    cursor.execute(
        """
        INSERT INTO dbo.APP_IncidentCaseDoctor (
            IncidentRequestCaseID,
            DoctorID,
            IsPrimary,
            AssignedByUserID
        )
        OUTPUT INSERTED.IncidentCaseDoctorID
        VALUES (?, ?, ?, ?)
        """,
        incident_id,
        doctor_id,
        int(is_primary),
        assigned_by_user_id,
    )

    incident_case_doctor_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()

    return incident_case_doctor_id


# -----------------------------
# READ
# -----------------------------

def list_case_doctors(incident_id: int) -> list[dict]:
    """
    List all doctors assigned to a case.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_IncidentCaseDoctor
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

def set_primary_doctor(
    incident_id: int,
    incident_case_doctor_id: int,
) -> None:
    """
    Set a specific doctor assignment as primary.
    Ensures only one primary doctor per case.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # Unset all
    cursor.execute(
        """
        UPDATE dbo.APP_IncidentCaseDoctor
        SET IsPrimary = 0
        WHERE IncidentRequestCaseID = ?
        """,
        incident_id,
    )

    # Set selected
    cursor.execute(
        """
        UPDATE dbo.APP_IncidentCaseDoctor
        SET IsPrimary = 1
        WHERE IncidentCaseDoctorID = ?
          AND IncidentRequestCaseID = ?
        """,
        incident_case_doctor_id,
        incident_id,
    )

    conn.commit()
    conn.close()


# -----------------------------
# DELETE
# -----------------------------

def remove_doctor_from_case(
    incident_case_doctor_id: int,
) -> None:
    """
    Remove a doctor assignment from a case.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        DELETE FROM dbo.APP_IncidentCaseDoctor
        WHERE IncidentCaseDoctorID = ?
        """,
        incident_case_doctor_id,
    )

    conn.commit()
    conn.close()
