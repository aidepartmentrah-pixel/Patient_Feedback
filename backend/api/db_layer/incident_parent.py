from typing import Optional, List
from core.database import get_connection
from backend.api.constants.case_statuses import PREPARATION_STATUSES, CLOSED_STATUS_ID


def create_incident_parent(data: dict) -> int:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        # incident_number is NOT NULL with no DB-side default. incident_id
        # isn't known until after the INSERT (it's an IDENTITY column), so
        # the real value can't be computed up front -- insert a placeholder
        # to satisfy the constraint, then overwrite it below once incident_id
        # is known. Same "INC-000123" format the /next-numbers preview
        # endpoint already shows workers before they save
        # (insert_router.py get_next_numbers).
        cursor.execute(
            """
            INSERT INTO dbo.APP_Incident (
                incident_number,
                patient_name,
                primary_doctor_name,
                primary_worker_name,
                feedback_intent_type_id,
                issuing_org_unit_id,
                complaint_summary,
                building_id,
                is_inpatient,
                created_by_user_id
            )
            OUTPUT INSERTED.incident_id
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            "PENDING",
            data.get("patient_name"),
            data.get("primary_doctor_name"),
            data.get("primary_worker_name"),
            data.get("feedback_intent_type_id"),
            data.get("issuing_org_unit_id"),
            data.get("complaint_summary"),
            data.get("building_id"),
            data.get("is_inpatient"),
            data.get("created_by_user_id"),
        )
        row = cursor.fetchone()
        incident_id = int(row[0])

        cursor.execute(
            "UPDATE dbo.APP_Incident SET incident_number = ? WHERE incident_id = ?",
            f"INC-{incident_id:06d}",
            incident_id,
        )
        conn.commit()
        return incident_id
    finally:
        conn.close()


def assign_case_to_incident(case_id: int, incident_id: int) -> None:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            UPDATE dbo.APP_IncidentCase
            SET incident_id = ?
            WHERE IncidentRequestCaseID = ?
            """,
            incident_id,
            case_id,
        )
        conn.commit()
    finally:
        conn.close()


def get_incident_parent(incident_id: int) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT
                i.incident_id,
                i.incident_number,
                i.patient_name,
                i.primary_doctor_name,
                i.primary_worker_name,
                i.feedback_intent_type_id,
                fit.NameEn AS feedback_intent_type_name,
                i.issuing_org_unit_id,
                org.Name AS issuing_org_unit_name,
                i.complaint_summary,
                i.building_id,
                b.BuildingName AS building_name,
                i.is_inpatient,
                i.created_at,
                i.created_by_user_id
            FROM dbo.APP_Incident i
            LEFT JOIN AdminsrationUnit org ON i.issuing_org_unit_id = org.UniqueID
            LEFT JOIN APP_LOOKUP_BUILDING b ON i.building_id = b.BuildingID
            LEFT JOIN APP_LOOKUP_FEEDBACK_INTENT_TYPE fit ON i.feedback_intent_type_id = fit.FeedbackIntentTypeID
            WHERE i.incident_id = ?
            """,
            incident_id,
        )
        row = cursor.fetchone()
        if not row:
            return None
        columns = [col[0] for col in cursor.description]
        incident = dict(zip(columns, row))

        # Format dates
        if incident.get("created_at"):
            incident["created_at"] = incident["created_at"].isoformat()

        # Fetch linked cases
        cursor.execute(
            """
            SELECT
                c.IncidentRequestCaseID AS case_id,
                CONCAT('REC-', YEAR(c.CreatedAt), '-', RIGHT(CONCAT('0000', c.IncidentRequestCaseID), 4)) AS record_id,
                c.CaseStatusID,
                cs.Name AS case_status_name,
                c.IssuingOrgUnitID AS target_org_unit_id,
                org.Name AS target_org_unit_name,
                c.DomainID AS domain_id,
                c.CategoryID AS category_id,
                c.SeverityID AS severity_id,
                c.FeedbackRecievedDate AS feedback_received_date
            FROM dbo.APP_IncidentCase c
            LEFT JOIN APP_LOOKUP_CASE_STATUS cs ON c.CaseStatusID = cs.CaseStatusID
            LEFT JOIN AdminsrationUnit org ON c.IssuingOrgUnitID = org.UniqueID
            WHERE c.incident_id = ?
            ORDER BY c.IncidentRequestCaseID
            """,
            incident_id,
        )
        case_rows = cursor.fetchall()
        case_cols = [col[0] for col in cursor.description]
        cases = []
        for r in case_rows:
            case = dict(zip(case_cols, r))
            if case.get("feedback_received_date"):
                case["feedback_received_date"] = case["feedback_received_date"].strftime("%Y-%m-%d")
            cases.append(case)

        incident["cases"] = cases
        return incident
    finally:
        conn.close()


def add_case_to_incident(incident_id: int, created_by_user_id: int) -> int:
    """
    Creates a new blank Draft case (CaseStatusID=4) linked to an existing incident.
    Copies patient_name, building, inpatient flag from the incident parent and
    source/dates from the incident's first existing case so the new case is valid.
    Returns the new IncidentRequestCaseID.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT
                i.patient_name,
                i.issuing_org_unit_id,
                i.building_id,
                i.is_inpatient,
                c.SourceID          AS source_id,
                c.FeedbackRecievedDate AS feedback_received_date,
                c.IncidentDate      AS incident_date
            FROM dbo.APP_Incident i
            OUTER APPLY (
                SELECT TOP 1 SourceID, FeedbackRecievedDate, IncidentDate
                FROM dbo.APP_IncidentCase
                WHERE incident_id = i.incident_id
                ORDER BY IncidentRequestCaseID
            ) c
            WHERE i.incident_id = ?
            """,
            (incident_id,),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Incident {incident_id} not found")

        cursor.execute(
            """
            INSERT INTO dbo.APP_IncidentCase (
                incident_id,
                CaseStatusID,
                RecordTypeID,
                ComplaintText,
                ImmediateAction,
                TakenAction,
                PatientName,
                IssuingOrgUnitID,
                BuildingID,
                isINPatient,
                SourceID,
                FeedbackRecievedDate,
                IncidentDate,
                ClinicalRiskTypeID,
                ExplanationStatusID,
                CreatedByUserID
            )
            OUTPUT INSERTED.IncidentRequestCaseID
            VALUES (?, 4, 1, '', '', '', ?, ?, ?, ?, ?, ?, ?, 1, 4, ?)
            """,
            (
                incident_id,
                row.patient_name,
                row.issuing_org_unit_id,
                row.building_id,
                row.is_inpatient,
                row.source_id,
                row.feedback_received_date,
                row.incident_date,
                created_by_user_id,
            ),
        )
        new_case_id = int(cursor.fetchone()[0])
        conn.commit()
        return new_case_id
    finally:
        conn.close()


def delete_case_from_incident(incident_id: int, case_id: int, deleted_by_user_id: int) -> dict:
    """
    Delete one case from a multi-case incident. Never allowed to remove an
    incident's last remaining case — there must always be at least one case
    left to trace the incident by.

    - Draft / Ready to Send cases (never published): hard-deleted outright,
      along with their doctor/target-department/employee/feedback/action-item
      rows, mirroring incident_case.hard_delete_incident_case()'s per-row
      cleanup but scoped to just this one case. Nothing has ever been sent
      anywhere, so there's no history to lose.
    - Any other status (already published): soft-deleted instead. The case
      row is kept (CaseStatusID set to Closed) and its subcase(s) are
      retired via administrative_subcase_db.retire_subcases_for_case (the
      same "excluded from every inbox query, row kept for audit" mechanism
      used when a published case's target unit changes), so it disappears
      from active workflow/reporting without destroying its history.

    Returns: {"deleted": True, "mode": "hard" | "soft"}
    Raises: ValueError if the case doesn't belong to this incident, or if
    it's the incident's only case.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "SELECT CaseStatusID FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ? AND incident_id = ?",
            (case_id, incident_id),
        )
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Case {case_id} does not belong to incident {incident_id}")
        case_status_id = row.CaseStatusID

        cursor.execute(
            "SELECT COUNT(*) FROM dbo.APP_IncidentCase WHERE incident_id = ?",
            (incident_id,),
        )
        total_cases = cursor.fetchone()[0]
        if total_cases <= 1:
            raise ValueError("Cannot delete the only case in an incident")

        is_unpublished = case_status_id in PREPARATION_STATUSES

        if is_unpublished:
            cursor.execute("DELETE FROM dbo.APP_IncidentCaseDoctor WHERE IncidentRequestCaseID = ?", (case_id,))
            cursor.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (case_id,))
            cursor.execute("DELETE FROM dbo.APP_IncidentCaseEmployee WHERE IncidentRequestCaseID = ?", (case_id,))
            cursor.execute("DELETE FROM dbo.APP_IncidentCaseFeedback WHERE IncidentRequestCaseID = ?", (case_id,))
            cursor.execute("DELETE FROM dbo.APP_ActionItem WHERE IncidentRequestCaseID = ?", (case_id,))
            cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        else:
            cursor.execute(
                "UPDATE dbo.APP_IncidentCase SET CaseStatusID = ?, UpdatedAt = GETDATE() WHERE IncidentRequestCaseID = ?",
                (CLOSED_STATUS_ID, case_id),
            )

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    if not is_unpublished:
        # Own connection, run after the case-row transaction commits — same
        # pattern as the retarget/auto-publish hooks in case_service.py.
        from backend.api_v2.db_layer import administrative_subcase_db
        administrative_subcase_db.retire_subcases_for_case(case_id, "CASE_DELETED", deleted_by_user_id)

    return {"deleted": True, "mode": "hard" if is_unpublished else "soft"}


def list_incidents(page: int = 1, page_size: int = 50, search: Optional[str] = None) -> dict:
    conn = get_connection()
    cursor = conn.cursor()
    try:
        where = ""
        params = []
        if search:
            where = "WHERE i.patient_name LIKE ? OR i.incident_number LIKE ?"
            s = f"%{search}%"
            params.extend([s, s])

        offset = (page - 1) * page_size

        count_query = f"SELECT COUNT(*) FROM dbo.APP_Incident i {where}"
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]

        query = f"""
            SELECT
                i.incident_id,
                i.incident_number,
                i.patient_name,
                i.feedback_intent_type_id,
                fit.NameEn AS feedback_intent_type_name,
                i.issuing_org_unit_id,
                org.Name AS issuing_org_unit_name,
                i.created_at,
                COUNT(c.IncidentRequestCaseID) AS case_count
            FROM dbo.APP_Incident i
            LEFT JOIN AdminsrationUnit org ON i.issuing_org_unit_id = org.UniqueID
            LEFT JOIN APP_LOOKUP_FEEDBACK_INTENT_TYPE fit ON i.feedback_intent_type_id = fit.FeedbackIntentTypeID
            LEFT JOIN dbo.APP_IncidentCase c ON c.incident_id = i.incident_id
            {where}
            GROUP BY
                i.incident_id, i.incident_number, i.patient_name,
                i.feedback_intent_type_id, fit.NameEn,
                i.issuing_org_unit_id, org.Name, i.created_at
            ORDER BY i.incident_id DESC
            OFFSET ? ROWS FETCH NEXT ? ROWS ONLY
        """
        params.extend([offset, page_size])
        cursor.execute(query, params)
        cols = [col[0] for col in cursor.description]
        incidents = []
        for row in cursor.fetchall():
            item = dict(zip(cols, row))
            if item.get("created_at"):
                item["created_at"] = item["created_at"].isoformat()
            incidents.append(item)

        return {
            "incidents": incidents,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_records": total,
                "total_pages": (total + page_size - 1) // page_size,
            },
        }
    finally:
        conn.close()
