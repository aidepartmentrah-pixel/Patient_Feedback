"""
Current Delay / Blockage Report Service (HCAT Performance & Delay Monitoring - Session 4)

LIVE operational report: for every org unit in scope, how many cases are
currently pending at that unit, how many of those are overdue, how old the
oldest/average pending case is, and the unit's historical force-close /
extra-time counts.

This is a CURRENT-STATE report. It does not use a date range - it reflects
"right now". It does not change or duplicate any Automatic Force Close
(Stage 1), Operational Dashboard Summary (Session 2), or Response
Performance Report (Session 3) logic - it only reads
APP_AdministrativeSubcase rows scoped by TargetOrgUnitID, the same scoping
field already used by those.
"""

from datetime import datetime
from core.database import get_connection
from ..constants.case_statuses import CLOSED_STATUS_ID
from api_v2.services.automatic_force_close_service import (
    _SECTION_PENDING_STATUSES,
    _DEPARTMENT_PENDING_STATUSES,
    _ADMINISTRATION_PENDING_STATUSES,
    _RECORD_TYPE_NOTICE,
)
from .response_performance_service import (
    _ORG_TYPE_ADMINISTRATION,
    _ORG_TYPE_SECTION,
    _ORG_TYPE_DEPARTMENT,
    _UNIT_TYPE_NAMES,
    LEVEL_TO_ORG_TYPE,
    _status_list_sql,
)


def get_current_delay_report(
    scope_unit_ids: list[int],
    level: str | None = None,
    target_unit_id: int | None = None,
) -> list[dict]:
    """
    Build the Current Delay / Blockage Report rows.

    Args:
        scope_unit_ids: org unit IDs the requesting user is allowed to see
            (already RBAC-intersected by the caller).
        level: "All" | "Administration" | "Department" | "Section" - filters
            rows to units of the matching AdminsrationUnit.Type. "All" or
            None means no level filter.
        target_unit_id: if given, restricts the report to this single unit
            (caller must have already verified it is in scope).

    Returns one row per (TargetOrgUnitID) that has at least one
    APP_AdministrativeSubcase row (CaseType='INCIDENT_RESPONSE') targeting it.
    There is no date range - this report reflects the CURRENT state.

    --------------------------------------------------------------------
    DOCUMENTED ASSUMPTIONS:

    - Per-unit "level": each subcase row's relevant Section/Department/
      Administration column-set is determined by the TARGET UNIT'S OWN
      AdminsrationUnit.Type (324/325/323) - same model as Session 3.

    - "Active case" filter (applied to Currently Pending, Currently Overdue,
      Oldest/Average Pending Days): excludes rows whose linked
      APP_IncidentCase is Closed (CaseStatusID = CLOSED_STATUS_ID), a Notice
      (RecordTypeID = 2), or Morbidity-related (IsMorbidity = 1). Seasonal-
      report subcases have no IncidentRequestCaseID (LEFT JOIN -> NULL on all
      three columns) and are treated as "active" (not Closed/Notice/Morbidity).

    - Currently Pending = COUNT of "active" rows whose Status is in this
      unit's level-specific pending-status tuple (the same
      _SECTION/_DEPARTMENT/_ADMINISTRATION_PENDING_STATUSES tuples used by
      the Automatic Force Close engine and Session 2/3). These tuples already
      include "returned for revision" statuses (the unit must act again) and
      escalation statuses such as FORCE_CLOSED_AT_SECTION /
      FORCE_CLOSED_AT_DEPARTMENT (the case has escalated TO this unit because
      the previous level was auto-force-closed) - both require action from
      THIS unit, and are excluded for the level that was force-closed (its
      Status is no longer in ITS OWN pending tuple).

    - Currently Overdue = Currently Pending rows whose this-level
      {Level}DeadlineAt is NOT NULL and < now.

    - Force Closed At Unit = {Level}ForceClosedAt IS NOT NULL (independent,
      historical flag - same as Session 3's "force_closed". Not subject to
      the Notice/Morbidity/Closed filter: the Automatic Force Close engine
      already never force-closes Notice or Morbidity cases, so this is moot
      in practice).

    - Extra Time Granted = {Level}ExtraTimeGrantedAt IS NOT NULL (independent,
      historical flag - same as Session 3's "extra_time_granted").

    - "Active responsibility start timestamp" (for Oldest/Average Pending
      Days) = COALESCE(UpdatedAt, CreatedAt). Every status transition
      (update_subcase_status, force_close_*, give_more_time) stamps
      UpdatedAt = now() AT THE MOMENT the row enters its current Status.
      Since "Currently Pending" is defined entirely by the row's CURRENT
      Status, COALESCE(UpdatedAt, CreatedAt) is exactly the timestamp this
      unit became responsible for the row:
        * Brand-new subcases (Status='SUBMITTED_TO_SECTION', never updated)
          have UpdatedAt = NULL, so CreatedAt (publish time) is used - correct,
          since Section became responsible at publish time.
        * Any other pending status (escalation, return-for-revision,
          give-more-time restart) was reached via an UPDATE that stamped
          UpdatedAt = now() at that exact transition.
      Pending Age (days) = DATEDIFF(day, COALESCE(UpdatedAt, CreatedAt), now).

    - Oldest Pending Days = MAX(Pending Age) over this unit's Currently
      Pending rows (0 if none). Average Pending Days = AVG(Pending Age) over
      the same set, rounded to 1 decimal (0.0 if none).
    --------------------------------------------------------------------
    """
    if not scope_unit_ids:
        return []

    unit_ids = [target_unit_id] if target_unit_id is not None else list(scope_unit_ids)
    if not unit_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    now = datetime.now()

    where_parts = ["s.CaseType = 'INCIDENT_RESPONSE'"]
    where_params: list = []

    placeholders = ",".join("?" * len(unit_ids))
    where_parts.append(f"s.TargetOrgUnitID IN ({placeholders})")
    where_params.extend(unit_ids)

    if level and level != "All":
        org_type = LEVEL_TO_ORG_TYPE.get(level)
        if org_type is not None:
            where_parts.append("u.Type = ?")
            where_params.append(org_type)

    where_clause = " AND ".join(where_parts)

    section_pending = _status_list_sql(_SECTION_PENDING_STATUSES)
    department_pending = _status_list_sql(_DEPARTMENT_PENDING_STATUSES)
    administration_pending = _status_list_sql(_ADMINISTRATION_PENDING_STATUSES)

    active_case = (
        f"(c.CaseStatusID IS NULL OR c.CaseStatusID != {CLOSED_STATUS_ID}) "
        f"AND (c.RecordTypeID IS NULL OR c.RecordTypeID != {_RECORD_TYPE_NOTICE}) "
        f"AND (c.IsMorbidity IS NULL OR c.IsMorbidity != 1)"
    )

    pending_case = f"""
        CASE
            WHEN u.Type = {_ORG_TYPE_SECTION} AND s.Status IN ({section_pending}) AND {active_case} THEN 1
            WHEN u.Type = {_ORG_TYPE_DEPARTMENT} AND s.Status IN ({department_pending}) AND {active_case} THEN 1
            WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} AND s.Status IN ({administration_pending}) AND {active_case} THEN 1
            ELSE 0
        END
    """

    age_days_case = f"""
        CASE
            WHEN u.Type = {_ORG_TYPE_SECTION} AND s.Status IN ({section_pending}) AND {active_case}
                THEN DATEDIFF(day, COALESCE(s.UpdatedAt, s.CreatedAt), ?)
            WHEN u.Type = {_ORG_TYPE_DEPARTMENT} AND s.Status IN ({department_pending}) AND {active_case}
                THEN DATEDIFF(day, COALESCE(s.UpdatedAt, s.CreatedAt), ?)
            WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} AND s.Status IN ({administration_pending}) AND {active_case}
                THEN DATEDIFF(day, COALESCE(s.UpdatedAt, s.CreatedAt), ?)
            ELSE NULL
        END
    """

    query = f"""
        SELECT
            u.UniqueID AS unit_id,
            u.Name AS unit_name,
            u.Type AS unit_type_code,
            SUM({pending_case}) AS currently_pending,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} AND s.Status IN ({section_pending}) AND {active_case}
                         AND s.SectionDeadlineAt IS NOT NULL AND s.SectionDeadlineAt < ? THEN 1
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} AND s.Status IN ({department_pending}) AND {active_case}
                         AND s.DepartmentDeadlineAt IS NOT NULL AND s.DepartmentDeadlineAt < ? THEN 1
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} AND s.Status IN ({administration_pending}) AND {active_case}
                         AND s.AdministrationDeadlineAt IS NOT NULL AND s.AdministrationDeadlineAt < ? THEN 1
                    ELSE 0
                END) AS currently_overdue,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN CASE WHEN s.SectionForceClosedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN CASE WHEN s.DepartmentForceClosedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN CASE WHEN s.AdministrationForceClosedAt IS NOT NULL THEN 1 ELSE 0 END
                    ELSE 0
                END) AS force_closed_at_unit,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN CASE WHEN s.SectionExtraTimeGrantedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN CASE WHEN s.DepartmentExtraTimeGrantedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN CASE WHEN s.AdministrationExtraTimeGrantedAt IS NOT NULL THEN 1 ELSE 0 END
                    ELSE 0
                END) AS extra_time_granted,
            MAX({age_days_case}) AS oldest_pending_days,
            AVG(CAST({age_days_case} AS FLOAT)) AS average_pending_days
        FROM dbo.APP_AdministrativeSubcase s
        JOIN dbo.AdminsrationUnit u ON u.UniqueID = s.TargetOrgUnitID
        LEFT JOIN dbo.APP_IncidentCase c ON s.IncidentRequestCaseID = c.IncidentRequestCaseID
        WHERE {where_clause}
        GROUP BY u.UniqueID, u.Name, u.Type
        ORDER BY u.Name
    """

    # Param order follows textual order of "?" placeholders:
    # currently_overdue (3x now) -> MAX age_days_case (3x now) ->
    # AVG age_days_case (3x now) -> WHERE clause (unit_ids [+ level])
    params = [now] * 9 + where_params

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        oldest = row.oldest_pending_days
        average = row.average_pending_days

        results.append({
            "unit_id": row.unit_id,
            "unit_name": row.unit_name,
            "unit_type": _UNIT_TYPE_NAMES.get(row.unit_type_code, "Unknown"),
            "currently_pending": row.currently_pending or 0,
            "currently_overdue": row.currently_overdue or 0,
            "force_closed_at_unit": row.force_closed_at_unit or 0,
            "extra_time_granted": row.extra_time_granted or 0,
            "oldest_pending_days": int(oldest) if oldest is not None else 0,
            "average_pending_days": round(average, 1) if average is not None else 0.0,
        })

    return results
