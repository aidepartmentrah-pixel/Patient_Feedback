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

def create_season_case(
    *,
    season_id: int,
    department_id: int,
    season_case_status_id: int,
    created_by_user_id: int,
    seasonal_report_text: str | None = None,
    seasonal_report_department_feedback: str | None = None,
) -> int:
    """
    Create a seasonal case for a department within a season.
    Returns SeasonCaseID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO dbo.APP_SeasonCase (
            SeasonID,
            DepartmentID,
            SeasonalReportText,
            SeasonalReportDepartmentFeedback,
            SeasonCaseStatusID,
            CreatedByUserID
        )
        OUTPUT INSERTED.SeasonCaseID
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        season_id,
        department_id,
        seasonal_report_text,
        seasonal_report_department_feedback,
        season_case_status_id,
        created_by_user_id,
    )

    season_case_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()

    return season_case_id


# -----------------------------
# READ
# -----------------------------

def get_season_case_by_id(season_case_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_SeasonCase
        WHERE SeasonCaseID = ?
        """,
        season_case_id,
    )

    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return dict(zip(columns, row)) if row else None


def list_season_cases(
    season_id: int | None = None,
    department_id: int | None = None,
) -> list[dict]:
    """
    List season cases, optionally filtered by season or department.
    """
    query = """
        SELECT *
        FROM dbo.APP_SeasonCase
        WHERE 1 = 1
    """
    params = []

    if season_id is not None:
        query += " AND SeasonID = ?"
        params.append(season_id)

    if department_id is not None:
        query += " AND DepartmentID = ?"
        params.append(department_id)

    query += " ORDER BY CreatedAt DESC"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query, params)

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return [dict(zip(columns, row)) for row in rows]


# -----------------------------
# UPDATE
# -----------------------------

def update_season_case(
    season_case_id: int,
    updates: dict,
) -> None:
    """
    Partial update for season case fields.
    """
    ALLOWED_FIELDS = {
        "SeasonalReportText",
        "SeasonalReportDepartmentFeedback",
        "SeasonCaseStatusID",
    }

    fields = []
    values = []

    for key, value in updates.items():
        if key not in ALLOWED_FIELDS:
            continue
        fields.append(f"{key} = ?")
        values.append(value)

    if not fields:
        return

    values.append(season_case_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        UPDATE dbo.APP_SeasonCase
        SET {", ".join(fields)}
        WHERE SeasonCaseID = ?
        """,
        values,
    )

    conn.commit()
    conn.close()
