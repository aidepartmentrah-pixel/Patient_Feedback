"""
Response Performance Report Service (HCAT Performance & Delay Monitoring - Session 3)

Historical management report measuring how well organizational units respond
to complaint workflow assignments: on-time / late / force-closed / extra-time /
still-open counts, average response time, response rate, late rate, and a
simple performance label.

This is a HISTORICAL report. It does not change or duplicate any Automatic
Force Close (Stage 1) or Operational Dashboard Summary (Session 2) logic -
it only reads APP_AdministrativeSubcase rows scoped by TargetOrgUnitID, the
same scoping field already used by get_operational_summary().
"""

from datetime import datetime
from core.database import get_connection
from api_v2.services.automatic_force_close_service import (
    _SECTION_PENDING_STATUSES,
    _DEPARTMENT_PENDING_STATUSES,
    _ADMINISTRATION_PENDING_STATUSES,
)

# AdminsrationUnit.Type codes
_ORG_TYPE_ADMINISTRATION = 323
_ORG_TYPE_SECTION = 324
_ORG_TYPE_DEPARTMENT = 325

_UNIT_TYPE_NAMES = {
    _ORG_TYPE_ADMINISTRATION: "Administration",
    _ORG_TYPE_SECTION: "Section",
    _ORG_TYPE_DEPARTMENT: "Department",
}

# "Hierarchy Level" filter -> AdminsrationUnit.Type
LEVEL_TO_ORG_TYPE = {
    "Administration": _ORG_TYPE_ADMINISTRATION,
    "Section": _ORG_TYPE_SECTION,
    "Department": _ORG_TYPE_DEPARTMENT,
}


def _status_list_sql(statuses: tuple) -> str:
    """
    Render a fixed tuple of status-code constants (imported from the
    Automatic Force Close engine, not user input) as a SQL IN(...) literal.
    """
    return ",".join("'{}'".format(s) for s in statuses)


def get_response_performance_report(
    scope_unit_ids: list[int],
    date_from: datetime,
    date_to_exclusive: datetime,
    level: str | None = None,
    target_unit_id: int | None = None,
) -> list[dict]:
    """
    Build the Response Performance Report rows.

    Args:
        scope_unit_ids: org unit IDs the requesting user is allowed to see
            (already RBAC-intersected by the caller).
        date_from: inclusive lower bound for APP_AdministrativeSubcase.CreatedAt
            ("assigned" timestamp - see Total Assigned Cases below).
        date_to_exclusive: exclusive upper bound for CreatedAt.
        level: "All" | "Administration" | "Department" | "Section" - filters
            rows to units of the matching AdminsrationUnit.Type. "All" or
            None means no level filter.
        target_unit_id: if given, restricts the report to this single unit
            (caller must have already verified it is in scope).

    Returns one row per (TargetOrgUnitID) that has at least one subcase
    matching the filters.

    --------------------------------------------------------------------
    DOCUMENTED ASSUMPTIONS (per Session 3 spec - "document the chosen
    source" / "document assumptions clearly"):

    - Total Assigned Cases: COUNT of APP_AdministrativeSubcase rows with
      CaseType='INCIDENT_RESPONSE', TargetOrgUnitID = unit, and CreatedAt
      in [date_from, date_to). Subcase rows are created exclusively by the
      publish flow (create_subcases_for_incident), so this set already
      excludes drafts and un-published "Ready To Send" cases - there is no
      separate "published" flag to check.

    - Per-unit "level": each subcase row's relevant Section/Department/
      Administration column-set is determined by the TARGET UNIT'S OWN
      AdminsrationUnit.Type (324/325/323), because create_subcases_for_incident
      sets exactly ONE of SectionDeadlineAt/DepartmentDeadlineAt/
      AdministrationDeadlineAt at publish time, matching that type. A unit's
      row therefore always reads its own {Level}* columns.

    - Answered On Time / Answered Late / Still Open are derived from three
      mutually-exclusive, exhaustive buckets per row (using the unit's level
      columns and the Automatic Force Close engine's per-level pending-status
      tuples):
        * Still Open      = LateReply=0 AND ForceClosedAt IS NULL
                            AND Status IN <level pending statuses>
        * Answered Late   = LateReply=1 (permanent historical flag - matches
                            Force Closed in practice, since AFC is currently
                            the only mechanism that sets LateReply)
        * Answered On Time = LateReply=0 AND ForceClosedAt IS NULL
                            AND Status NOT IN <level pending statuses>
      total_assigned == still_open + answered_on_time + answered_late.

    - Force Closed = {Level}ForceClosedAt IS NOT NULL.
    - Extra Time Granted = {Level}ExtraTimeGrantedAt IS NOT NULL.

    - Average Response Days = AVG(DATEDIFF(day, CreatedAt,
      COALESCE({Level}ForceClosedAt, UpdatedAt, CreatedAt))), computed only
      over "answered" rows (on-time + late), i.e. excluding Still Open.
      KNOWN LIMITATION: normal (non-force-closed) workflow transitions do not
      record a dedicated "{Level} responded at" timestamp - update_subcase_status
      only stamps UpdatedAt at the time of the LATEST status change on the row.
      For a row that has progressed past this unit's level, UpdatedAt may
      reflect a later level's action, slightly overestimating this unit's own
      response time. UpdatedAt/ForceClosedAt are the closest reliable sources
      available without a schema change (flagged as a Session 4 risk).

    - Response Rate = (Answered On Time + Answered Late) / Total Assigned * 100
    - Late Rate     = Answered Late / Total Assigned * 100
    - Performance Label: Good (late_rate <= 10), Warning (10 < late_rate <= 25),
      Poor (late_rate > 25).
    --------------------------------------------------------------------
    """
    if not scope_unit_ids:
        return []

    unit_ids = [target_unit_id] if target_unit_id is not None else list(scope_unit_ids)
    if not unit_ids:
        return []

    conn = get_connection()
    cursor = conn.cursor()

    where_parts = [
        "s.CaseType = 'INCIDENT_RESPONSE'",
        "s.CreatedAt >= ?",
        "s.CreatedAt < ?",
    ]
    params: list = [date_from, date_to_exclusive]

    placeholders = ",".join("?" * len(unit_ids))
    where_parts.append(f"s.TargetOrgUnitID IN ({placeholders})")
    params.extend(unit_ids)

    if level and level != "All":
        org_type = LEVEL_TO_ORG_TYPE.get(level)
        if org_type is not None:
            where_parts.append("u.Type = ?")
            params.append(org_type)

    where_clause = " AND ".join(where_parts)

    section_pending = _status_list_sql(_SECTION_PENDING_STATUSES)
    department_pending = _status_list_sql(_DEPARTMENT_PENDING_STATUSES)
    administration_pending = _status_list_sql(_ADMINISTRATION_PENDING_STATUSES)

    query = f"""
        SELECT
            u.UniqueID AS unit_id,
            u.Name AS unit_name,
            u.Type AS unit_type_code,
            COUNT(*) AS total_assigned,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN
                        CASE WHEN s.SectionLateReply = 0 AND s.SectionForceClosedAt IS NULL
                              AND s.Status NOT IN ({section_pending}) THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN
                        CASE WHEN s.DepartmentLateReply = 0 AND s.DepartmentForceClosedAt IS NULL
                              AND s.Status NOT IN ({department_pending}) THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN
                        CASE WHEN s.AdministrationLateReply = 0 AND s.AdministrationForceClosedAt IS NULL
                              AND s.Status NOT IN ({administration_pending}) THEN 1 ELSE 0 END
                    ELSE 0
                END) AS answered_on_time,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN CASE WHEN s.SectionLateReply = 1 THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN CASE WHEN s.DepartmentLateReply = 1 THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN CASE WHEN s.AdministrationLateReply = 1 THEN 1 ELSE 0 END
                    ELSE 0
                END) AS answered_late,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN CASE WHEN s.SectionForceClosedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN CASE WHEN s.DepartmentForceClosedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN CASE WHEN s.AdministrationForceClosedAt IS NOT NULL THEN 1 ELSE 0 END
                    ELSE 0
                END) AS force_closed,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN CASE WHEN s.SectionExtraTimeGrantedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN CASE WHEN s.DepartmentExtraTimeGrantedAt IS NOT NULL THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN CASE WHEN s.AdministrationExtraTimeGrantedAt IS NOT NULL THEN 1 ELSE 0 END
                    ELSE 0
                END) AS extra_time_granted,
            SUM(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION} THEN
                        CASE WHEN s.SectionLateReply = 0 AND s.SectionForceClosedAt IS NULL
                              AND s.Status IN ({section_pending}) THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT} THEN
                        CASE WHEN s.DepartmentLateReply = 0 AND s.DepartmentForceClosedAt IS NULL
                              AND s.Status IN ({department_pending}) THEN 1 ELSE 0 END
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION} THEN
                        CASE WHEN s.AdministrationLateReply = 0 AND s.AdministrationForceClosedAt IS NULL
                              AND s.Status IN ({administration_pending}) THEN 1 ELSE 0 END
                    ELSE 0
                END) AS still_open,
            AVG(CASE
                    WHEN u.Type = {_ORG_TYPE_SECTION}
                         AND NOT (s.SectionLateReply = 0 AND s.SectionForceClosedAt IS NULL
                                  AND s.Status IN ({section_pending}))
                        THEN CAST(DATEDIFF(day, s.CreatedAt, COALESCE(s.SectionForceClosedAt, s.UpdatedAt, s.CreatedAt)) AS FLOAT)
                    WHEN u.Type = {_ORG_TYPE_DEPARTMENT}
                         AND NOT (s.DepartmentLateReply = 0 AND s.DepartmentForceClosedAt IS NULL
                                  AND s.Status IN ({department_pending}))
                        THEN CAST(DATEDIFF(day, s.CreatedAt, COALESCE(s.DepartmentForceClosedAt, s.UpdatedAt, s.CreatedAt)) AS FLOAT)
                    WHEN u.Type = {_ORG_TYPE_ADMINISTRATION}
                         AND NOT (s.AdministrationLateReply = 0 AND s.AdministrationForceClosedAt IS NULL
                                  AND s.Status IN ({administration_pending}))
                        THEN CAST(DATEDIFF(day, s.CreatedAt, COALESCE(s.AdministrationForceClosedAt, s.UpdatedAt, s.CreatedAt)) AS FLOAT)
                    ELSE NULL
                END) AS avg_response_days
        FROM dbo.APP_AdministrativeSubcase s
        JOIN dbo.AdminsrationUnit u ON u.UniqueID = s.TargetOrgUnitID
        WHERE {where_clause}
        GROUP BY u.UniqueID, u.Name, u.Type
        ORDER BY u.Name
    """

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        total = row.total_assigned or 0
        on_time = row.answered_on_time or 0
        late = row.answered_late or 0
        force_closed = row.force_closed or 0
        extra_time = row.extra_time_granted or 0
        still_open = row.still_open or 0
        avg_days = row.avg_response_days

        answered = on_time + late
        response_rate = round((answered / total * 100), 1) if total else 0.0
        late_rate = round((late / total * 100), 1) if total else 0.0

        if late_rate <= 10:
            label = "Good"
        elif late_rate <= 25:
            label = "Warning"
        else:
            label = "Poor"

        results.append({
            "unit_id": row.unit_id,
            "unit_name": row.unit_name,
            "unit_type": _UNIT_TYPE_NAMES.get(row.unit_type_code, "Unknown"),
            "total_assigned": total,
            "answered_on_time": on_time,
            "answered_late": late,
            "force_closed": force_closed,
            "extra_time_granted": extra_time,
            "still_open": still_open,
            "average_response_days": round(avg_days, 1) if avg_days is not None else 0.0,
            "response_rate": response_rate,
            "late_rate": late_rate,
            "performance_label": label,
        })

    return results
