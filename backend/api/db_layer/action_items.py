from datetime import date
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


def create_action_item(
    *,
    action_title: str,
    created_by_user_id: int,
    incident_case_id: int | None = None,
    seasonal_report_id: int | None = None,
    season_case_id: int | None = None,
    action_description: str | None = None,
    due_date: date | None = None,
) -> int:
    """
    Create an action item linked to exactly ONE parent:
    - Incident case (IncidentRequestCaseID)
    - Seasonal report (SeasonalReportID)
    - Season case (SeasonCaseID)
    
    NOTE: Validation that exactly one parent is provided should be done
    in the service layer before calling this function.
    
    Returns ActionItemID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO dbo.APP_ActionItem (
            IncidentRequestCaseID,
            SeasonalReportID,
            SeasonCaseID,
            ActionTitle,
            ActionDescription,
            DueDate,
            CreatedByUserID
        )
        OUTPUT INSERTED.ActionItemID
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        incident_case_id,
        seasonal_report_id,
        season_case_id,
        action_title,
        action_description,
        due_date,
        created_by_user_id,
    )

    action_item_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()

    return action_item_id


# -----------------------------
# READ
# -----------------------------

def get_action_item_by_id(action_item_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_ActionItem
        WHERE ActionItemID = ?
        """,
        action_item_id,
    )

    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return dict(zip(columns, row)) if row else None

def list_action_items_for_incident(incident_case_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_ActionItem
        WHERE IncidentRequestCaseID = ?
        ORDER BY CreatedAt DESC
        """,
        incident_case_id,
    )

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return [dict(zip(columns, row)) for row in rows]

def list_action_items_for_season(season_case_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_ActionItem
        WHERE SeasonCaseID = ?
        ORDER BY CreatedAt DESC
        """,
        season_case_id,
    )

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return [dict(zip(columns, row)) for row in rows]

def list_action_items_for_seasonal_report(seasonal_report_id: int) -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM dbo.APP_ActionItem
        WHERE SeasonalReportID = ?
        ORDER BY CreatedAt DESC
        """,
        seasonal_report_id,
    )

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return [dict(zip(columns, row)) for row in rows]

def update_action_item(
    action_item_id: int,
    updates: dict,
) -> None:
    """
    Partial update for editable fields.
    """
    ALLOWED_FIELDS = {
        "ActionTitle",
        "ActionDescription",
        "DueDate",
        "IsDone",
        "DateSubmitted",
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

    values.append(action_item_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        UPDATE dbo.APP_ActionItem
        SET {", ".join(fields)}
        WHERE ActionItemID = ?
        """,
        values,
    )

    conn.commit()
    conn.close()

def mark_action_item_done(action_item_id: int) -> None:
    """
    Mark an action item as completed.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE dbo.APP_ActionItem
        SET IsDone = 1,
            DateSubmitted = CAST(GETDATE() AS date)
        WHERE ActionItemID = ?
        """,
        action_item_id,
    )

    conn.commit()
    conn.close()