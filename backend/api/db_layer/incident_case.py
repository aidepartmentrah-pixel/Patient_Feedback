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
# ALLOWED UPDATE FIELDS
# -----------------------------

UPDATABLE_FIELDS = {
    "ComplaintText",
    "ImmediateAction",
    "TakenAction",
    "FeedbackRecievedDate",
    "PatientName",
    "isINPatient",
    "ClinicalRiskTypeID",
    "FeedbackIntentTypeID",
    "BuildingID",
    "DomainID",
    "CategoryID",
    "SubCategoryID",
    "ClassificationID",
    "SeverityID",
    "StageID",
    "HarmLevelID",
    "CaseStatusID",
    "SourceID",
    "RequiresExplanation",
}


# -----------------------------
# CREATE
# -----------------------------

def create_incident_case(data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO dbo.APP_IncidentCase (
            ComplaintText,
            ImmediateAction,
            TakenAction,
            FeedbackRecievedDate,
            PatientName,
            IssuingOrgUnitID,
            CreatedByUserID,
            isINPatient,
            ClinicalRiskTypeID,
            FeedbackIntentTypeID,
            BuildingID,
            DomainID,
            CategoryID,
            SubCategoryID,
            ClassificationID,
            SeverityID,
            StageID,
            HarmLevelID,
            CaseStatusID,
            SourceID,
            ExplanationStatusID,
            RequiresExplanation
        )
        OUTPUT INSERTED.IncidentRequestCaseID
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        data.get("ComplaintText"),
        data.get("ImmediateAction") or "",
        data.get("TakenAction") or "",
        data.get("FeedbackRecievedDate"),
        data.get("PatientName") or "",
        data.get("IssuingOrgUnitID"),
        data.get("CreatedByUserID"),
        data.get("isINPatient", 1),
        data.get("ClinicalRiskTypeID"),
        data.get("FeedbackIntentTypeID"),
        data.get("BuildingID"),
        data.get("DomainID"),
        data.get("CategoryID"),
        data.get("SubCategoryID"),
        data.get("ClassificationID"),
        data.get("SeverityID"),
        data.get("StageID"),
        data.get("HarmLevelID"),
        data.get("CaseStatusID"),
        data.get("SourceID"),
        data.get("ExplanationStatusID"),
        data.get("RequiresExplanation", 0),
    )

    incident_id = cursor.fetchone()[0]
    conn.commit()
    conn.close()
    return incident_id


# -----------------------------
# READ
# -----------------------------

def get_incident_case_by_id(incident_id: int) -> dict | None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
        incident_id,
    )

    row = cursor.fetchone()
    columns = [col[0] for col in cursor.description]

    conn.close()

    return dict(zip(columns, row)) if row else None


def list_incident_cases() -> list[dict]:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT * FROM dbo.APP_IncidentCase ORDER BY CreatedAt DESC"
    )

    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]

    conn.close()

    return [dict(zip(columns, row)) for row in rows]


# -----------------------------
# UPDATE
# -----------------------------

def update_incident_case(incident_id: int, updates: dict) -> None:
    fields = []
    values = []

    for key, value in updates.items():
        if key not in UPDATABLE_FIELDS:
            continue
        fields.append(f"{key} = ?")
        values.append(value)

    if not fields:
        return

    values.append(incident_id)

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        UPDATE dbo.APP_IncidentCase
        SET {", ".join(fields)}
        WHERE IncidentRequestCaseID = ?
        """,
        values,
    )

    conn.commit()
    conn.close()


# -----------------------------
# SOFT DELETE
# -----------------------------

def soft_delete_incident_case(
    incident_id: int,
    closed_status_id: int,
) -> None:
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE dbo.APP_IncidentCase
        SET CaseStatusID = ?
        WHERE IncidentRequestCaseID = ?
        """,
        closed_status_id,
        incident_id,
    )

    conn.commit()
    conn.close()

from backend.core.database import get_connection


def hard_delete_incident_case(incident_id: int) -> None:
    """
    Completely and permanently deletes an incident case and ALL its related records.

    This will remove:
    - APP_IncidentCaseDoctor
    - APP_IncidentCaseTargetDepartment
    - APP_IncidentCaseFeedback
    - APP_ActionItem
    - APP_IncidentCase (parent)

    Uses a transaction for safety.
    """

    conn = get_connection()
    cursor = conn.cursor()

    try:
        # -----------------------------
        # Start transaction
        # -----------------------------
        conn.autocommit = False

        # -----------------------------
        # Delete children FIRST
        # -----------------------------

        # Doctors linked to case
        cursor.execute(
            "DELETE FROM dbo.APP_IncidentCaseDoctor WHERE IncidentRequestCaseID = ?",
            incident_id,
        )

        # Target departments
        cursor.execute(
            "DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?",
            incident_id,
        )

        # Feedback (1-to-1)
        cursor.execute(
            "DELETE FROM dbo.APP_IncidentCaseFeedback WHERE IncidentRequestCaseID = ?",
            incident_id,
        )

        # Action items
        cursor.execute(
            "DELETE FROM dbo.APP_ActionItem WHERE IncidentRequestCaseID = ?",
            incident_id,
        )

        # -----------------------------
        # Now delete the parent
        # -----------------------------
        cursor.execute(
            "DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?",
            incident_id,
        )

        # -----------------------------
        # Commit transaction
        # -----------------------------
        conn.commit()

    except Exception as e:
        conn.rollback()
        raise e

    finally:
        conn.close()
