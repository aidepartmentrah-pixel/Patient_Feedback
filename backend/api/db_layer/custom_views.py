from typing import Any, Dict, List, Optional
from core.database import get_connection


COLUMNS = [
    "ViewID",
    "ViewName",
    "ShowIncidentNumber",
    "ShowIncidentRequestCaseID",
    "ShowComplaintText",
    "ShowImmediateAction",
    "ShowTakenAction",
    "ShowFeedbackRecievedDate",
    "ShowPatientName",
    "ShowIssuingOrgUnitID",
    "ShowCreatedAt",
    "ShowCreatedByUserID",
    "ShowIsInPatient",
    "ShowClinicalRiskTypeID",
    "ShowFeedbackIntentTypeID",
    "ShowBuildingID",
    "ShowDomainID",
    "ShowCategoryID",
    "ShowSubCategoryID",
    "ShowClassificationID",
    "ShowSeverityID",
    "ShowStageID",
    "ShowHarmLevelID",
    "ShowCaseStatusID",
    "ShowSourceID",
    "ShowExplanationStatusID",
    "ShowSectionAnswer",
    "ShowDepartmentAnswer",
    "ShowAdministrationAnswer",
    "CreatedAt",
    "CreatedByUserID",
    "IsActive",
]

SHOW_COLUMNS = [c for c in COLUMNS if c.startswith("Show")]  # all Show* flags


def create_custom_view(data: Dict[str, Any]) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        INSERT INTO dbo.APP_CUSTOM_VIEWS (
            ViewName,
            {', '.join(SHOW_COLUMNS)},
            CreatedAt,
            CreatedByUserID,
            IsActive
        )
        OUTPUT INSERTED.ViewID
        VALUES (
            ?,
            {', '.join(['?' for _ in SHOW_COLUMNS])},
            ?, ?, ?
        )
        """,
        data.get("ViewName"),
        *[int(bool(data.get(col, False))) for col in SHOW_COLUMNS],
        data.get("CreatedAt"),
        data.get("CreatedByUserID"),
        int(bool(data.get("IsActive", True))),
    )

    view_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return view_id


def get_custom_view(view_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM dbo.APP_CUSTOM_VIEWS WHERE ViewID = ?",
        view_id,
    )
    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description] if cursor.description else []

    conn.close()
    return dict(zip(columns, row)) if row else None


def list_custom_views(active_only: bool = True) -> List[Dict[str, Any]]:
    conn = get_connection()
    cursor = conn.cursor()

    if active_only:
        cursor.execute(
            "SELECT * FROM dbo.APP_CUSTOM_VIEWS WHERE IsActive = 1 ORDER BY CreatedAt DESC"
        )
    else:
        cursor.execute(
            "SELECT * FROM dbo.APP_CUSTOM_VIEWS ORDER BY CreatedAt DESC"
        )

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()
    return [dict(zip(columns, row)) for row in rows]


UPDATABLE_FIELDS = set([
    "ViewName",
    *SHOW_COLUMNS,
    "IsActive",
])


def update_custom_view(view_id: int, updates: Dict[str, Any]) -> None:
    fields = []
    values = []

    for key, value in updates.items():
        if key not in UPDATABLE_FIELDS:
            continue
        fields.append(f"{key} = ?")
        if key.startswith("Show") or key == "IsActive":
            values.append(int(bool(value)))
        else:
            values.append(value)

    if not fields:
        return

    values.append(view_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"UPDATE dbo.APP_CUSTOM_VIEWS SET {', '.join(fields)} WHERE ViewID = ?",
        values,
    )

    conn.commit()
    conn.close()


def deactivate_custom_view(view_id: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "UPDATE dbo.APP_CUSTOM_VIEWS SET IsActive = 0 WHERE ViewID = ?",
        view_id,
    )

    conn.commit()
    conn.close()


def delete_custom_view(view_id: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM d  bo.APP_CUSTOM_VIEWS WHERE ViewID = ?",
        view_id,
    )

    conn.commit()
    conn.close()
