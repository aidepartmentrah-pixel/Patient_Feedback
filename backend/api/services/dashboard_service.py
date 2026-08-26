from datetime import date, datetime, timedelta
from collections import Counter, defaultdict
from ..db_layer import incident_case, admin_units, lookups
from ..schemas.auth_models import CurrentUser
from ..constants.case_statuses import OPEN_STATUS_ID, IN_PROGRESS_STATUS_ID, CLOSED_STATUS_ID
from ..constants.org_unit_types import ORG_TYPE_ADMINISTRATION
from . import org_tree_service

# =========================================================
# PUBLIC SERVICE FUNCTIONS
# =========================================================

def _row_to_dict(u):
    """
    Normalize DB row or dict into a dict.
    """
    if isinstance(u, dict):
        return u

    return {
        "UniqueID": u.UniqueID,
        "ParentID": u.ParentID,
        "Type": u.Type,
        "Name": u.Name,
        "Frozen": u.Frozen,
    }


def get_dashboard_stats(
    *,
    current_user: CurrentUser,
    scope: str,
    administration_id: int | None,
    department_id: int | None,
    section_id: int | None,
    start_date: date,
    end_date: date,
    classification_chart_type: str = "bar",
    stage_chart_type: str = "bar",
    department_chart_type: str = "bar",
) -> dict:
    """
    Core dashboard service.
    Returns metrics, trends, charts, recent activity.
    
    Dashboard Scope Filtering (Fixed):
    - Respects client-requested scope parameters (section/department/administration/hospital)
    - Expands hierarchical scopes using org_tree_service.get_descendants()
    - Intersects requested scope with current_user.allowed_unit_ids for RBAC safety
    - Router guards have already validated that requested IDs are accessible
    """

    # -------------------------
    # Determine Requested Scope
    # -------------------------
    # Build the set of org unit IDs based on what the client requested
    if scope == "section" and section_id is not None:
        # Section scope: just the single section
        requested_unit_ids = {section_id}
    
    elif scope == "department" and department_id is not None:
        # Department scope: department + all descendant sections
        requested_unit_ids = org_tree_service.get_descendants(department_id)
    
    elif scope == "administration" and administration_id is not None:
        # Administration scope: administration + all departments + all sections
        requested_unit_ids = org_tree_service.get_descendants(administration_id)
    
    else:
        # Hospital scope (or fallback): use user's full allowed scope
        requested_unit_ids = current_user.allowed_unit_ids
    
    # -------------------------
    # RBAC Safety: Intersect with allowed scope
    # -------------------------
    # Only include units the user is actually allowed to access
    scope_unit_ids = list(requested_unit_ids & current_user.allowed_unit_ids)
    
    # Determine if we should include issuing department chart
    # Show department breakdown for hospital/administration/department scopes
    include_issuing_dept = scope in ("hospital", "administration", "department")

    incidents = _fetch_incidents_in_scope(scope_unit_ids, start_date, end_date)
    
    # Count force-closed subcases from APP_AdministrativeSubcase table
    force_closed_count = _count_force_closed_subcases(scope_unit_ids, start_date, end_date)

    metrics = _compute_metrics(incidents, force_closed_subcase_count=force_closed_count)
    charts = _build_charts(
        incidents, 
        include_issuing_dept,
        classification_chart_type=classification_chart_type,
        stage_chart_type=stage_chart_type,
        department_chart_type=department_chart_type,
    )
    recent_activity = _build_recent_activity(incidents)

    previous_start = start_date - (end_date - start_date)
    previous_end = start_date

    previous_incidents = _fetch_incidents_in_scope(
        scope_unit_ids,
        previous_start,
        previous_end,
    )
    
    # Count force-closed subcases for previous period (for trend calculation)
    previous_force_closed = _count_force_closed_subcases(scope_unit_ids, previous_start, previous_end)

    trends = _compute_trends(
        _compute_metrics(previous_incidents, force_closed_subcase_count=previous_force_closed),
        metrics,
    )

    return {
        "metrics": metrics,
        "trends": trends,
        "charts": charts,
        "recentActivity": recent_activity,
    }


def get_dashboard_date_bounds_for_units(unit_ids: list[int]) -> dict:
    """
    Get minimum and maximum incident CreatedAt dates for given unit IDs.
    
    Supports frontend timeline slider date range selector.
    
    Args:
        unit_ids: List of organizational unit IDs to query
    
    Returns:
        dict with keys:
            - min_date: date object or None
            - max_date: date object or None
    """
    from core.database import get_connection
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Build WHERE clause — scoped by TargetOrgUnitID (see _fetch_incidents_in_scope)
    where_parts = []
    params = []

    if unit_ids:
        placeholders = ','.join('?' * len(unit_ids))
        where_parts.append(
            f"""EXISTS (
                SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment td
                WHERE td.IncidentRequestCaseID = c.IncidentRequestCaseID
                  AND td.DepartmentID IN ({placeholders})
            )"""
        )
        params.extend(unit_ids)

    where_clause = " AND ".join(where_parts) if where_parts else "1=1"

    # Query MIN/MAX dates using DATE cast
    query = f"""
        SELECT
            MIN(CAST(c.CreatedAt AS DATE)) AS min_date,
            MAX(CAST(c.CreatedAt AS DATE)) AS max_date
        FROM dbo.APP_IncidentCase c
        WHERE {where_clause}
    """
    
    cursor.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    
    # -------------------------
    # NULL SAFETY CONTRACT (DR-B3)
    # Always return dict with both keys
    # Values are date objects or None
    # -------------------------
    if not row or (row[0] is None and row[1] is None):
        # No incidents in scope - return explicit None values
        return {
            "min_date": None,
            "max_date": None
        }
    
    # Return date objects or None
    return {
        "min_date": row[0] if row[0] else None,
        "max_date": row[1] if row[1] else None
    }


def get_operational_summary(scope_unit_ids: list[int]) -> dict:
    """
    Operational dashboard summary (HCAT Performance & Delay Monitoring - Session 2).

    Single source of truth for the 6 operational dashboard cards. All counts
    are scoped to scope_unit_ids (already RBAC-intersected by the caller).

    Sources:
    - open_cases / closed_cases: APP_IncidentCase.CaseStatusID, scoped by
      TargetOrgUnitID via APP_IncidentCaseTargetDepartment (same scoping used
      by _fetch_incidents_in_scope / table_view_service — "cases targeted at
      this unit", not "cases this unit filed").
    - force_closed_cases: APP_AdministrativeSubcase rows where any of
      Section/Department/AdministrationForceClosedAt IS NOT NULL. These
      columns are set once (idempotency-guarded) and never cleared, so this
      count remains stable as status transitions evolve.
    - late_replies: APP_AdministrativeSubcase rows where any of
      Section/Department/AdministrationLateReply = 1. Permanent historical
      flags - never cleared by extra time grants or workflow completion.
    - extra_time_granted: APP_AdministrativeSubcase rows where any of
      Section/Department/AdministrationExtraTimeGrantedAt IS NOT NULL.
      Permanent historical record.
    - currently_overdue: APP_AdministrativeSubcase rows whose responsible
      level's deadline has passed while that level is still pending, using
      the same per-level pending-status sets as the Automatic Force Close
      engine. Excludes Notice cases, Morbidity cases, and Closed cases.
    All counts in this summary are now consistently scoped by TargetOrgUnitID
    (same scoping field used by _count_force_closed_subcases).
    """
    empty_result = {
        "open_cases": 0,
        "closed_cases": 0,
        "force_closed_cases": 0,
        "late_replies": 0,
        "currently_overdue": 0,
        "extra_time_granted": 0,
    }

    if not scope_unit_ids:
        return empty_result

    from core.database import get_connection
    from api_v2.services.automatic_force_close_service import (
        _SECTION_PENDING_STATUSES,
        _DEPARTMENT_PENDING_STATUSES,
        _ADMINISTRATION_PENDING_STATUSES,
        _RECORD_TYPE_NOTICE,
    )

    conn = get_connection()
    cursor = conn.cursor()

    placeholders = ','.join('?' * len(scope_unit_ids))

    # -------------------------
    # Open / Closed cases (APP_IncidentCase, scoped by TargetOrgUnitID via
    # APP_IncidentCaseTargetDepartment)
    # -------------------------
    cursor.execute(
        f"""
        SELECT
            SUM(CASE WHEN c.CaseStatusID IN (?, ?) THEN 1 ELSE 0 END) AS open_cases,
            SUM(CASE WHEN c.CaseStatusID = ? THEN 1 ELSE 0 END) AS closed_cases
        FROM dbo.APP_IncidentCase c
        WHERE EXISTS (
            SELECT 1 FROM dbo.APP_IncidentCaseTargetDepartment td
            WHERE td.IncidentRequestCaseID = c.IncidentRequestCaseID
              AND td.DepartmentID IN ({placeholders})
        )
        """,
        [OPEN_STATUS_ID, IN_PROGRESS_STATUS_ID, CLOSED_STATUS_ID, *scope_unit_ids],
    )
    row = cursor.fetchone()
    open_cases = row[0] or 0
    closed_cases = row[1] or 0

    # -------------------------
    # Force closed / late replies / extra time granted
    # (APP_AdministrativeSubcase, scoped by TargetOrgUnitID)
    # -------------------------
    cursor.execute(
        f"""
        SELECT
            SUM(CASE WHEN SectionForceClosedAt IS NOT NULL
                      OR DepartmentForceClosedAt IS NOT NULL
                      OR AdministrationForceClosedAt IS NOT NULL
                     THEN 1 ELSE 0 END) AS force_closed_cases,
            SUM(CASE WHEN SectionLateReply = 1
                      OR DepartmentLateReply = 1
                      OR AdministrationLateReply = 1
                     THEN 1 ELSE 0 END) AS late_replies,
            SUM(CASE WHEN SectionExtraTimeGrantedAt IS NOT NULL
                      OR DepartmentExtraTimeGrantedAt IS NOT NULL
                      OR AdministrationExtraTimeGrantedAt IS NOT NULL
                     THEN 1 ELSE 0 END) AS extra_time_granted
        FROM dbo.APP_AdministrativeSubcase
        WHERE TargetOrgUnitID IN ({placeholders})
        """,
        list(scope_unit_ids),
    )
    row = cursor.fetchone()
    force_closed_cases = row[0] or 0
    late_replies = row[1] or 0
    extra_time_granted = row[2] or 0

    # -------------------------
    # Currently overdue: one query per responsible level, summed.
    # Pending-status sets are mutually exclusive across levels, so no
    # subcase is counted twice. Excludes Notice/Morbidity/Closed cases.
    # -------------------------
    now = datetime.now()
    currently_overdue = 0

    for deadline_column, pending_statuses in (
        ("SectionDeadlineAt", _SECTION_PENDING_STATUSES),
        ("DepartmentDeadlineAt", _DEPARTMENT_PENDING_STATUSES),
        ("AdministrationDeadlineAt", _ADMINISTRATION_PENDING_STATUSES),
    ):
        status_placeholders = ','.join('?' * len(pending_statuses))
        cursor.execute(
            f"""
            SELECT COUNT(*)
            FROM dbo.APP_AdministrativeSubcase s
            LEFT JOIN dbo.APP_IncidentCase c
                ON s.IncidentRequestCaseID = c.IncidentRequestCaseID
            WHERE s.TargetOrgUnitID IN ({placeholders})
              AND s.{deadline_column} IS NOT NULL
              AND s.{deadline_column} < ?
              AND s.Status IN ({status_placeholders})
              AND (c.RecordTypeID IS NULL OR c.RecordTypeID != ?)
              AND (c.IsMorbidity IS NULL OR c.IsMorbidity != 1)
              AND (c.CaseStatusID IS NULL OR c.CaseStatusID != ?)
            """,
            [*scope_unit_ids, now, *pending_statuses, _RECORD_TYPE_NOTICE, CLOSED_STATUS_ID],
        )
        currently_overdue += cursor.fetchone()[0] or 0

    conn.close()

    return {
        "open_cases": open_cases,
        "closed_cases": closed_cases,
        "force_closed_cases": force_closed_cases,
        "late_replies": late_replies,
        "currently_overdue": currently_overdue,
        "extra_time_granted": extra_time_granted,
    }


def get_dashboard_hierarchy(current_user: CurrentUser) -> dict:
    """
    Returns organizational hierarchy for dashboard selectors
    filtered by user's allowed organizational scope.
    
    Phase 2.5: Only returns org units within current_user.allowed_unit_ids.

    Hierarchy rules (as implemented in DB):
    - Administration: ParentID == UniqueID
    - Department: ParentID == Administration.UniqueID AND UniqueID != ParentID
    - Section: ParentID == Department.UniqueID
    """

    raw_units = admin_units.get_admin_unit_tree()
    units = [_row_to_dict(u) for u in raw_units]

    # Exclude frozen (retired/test) units so they never leak into the
    # Administration/Department/Section selectors.
    units = [u for u in units if not u.get("Frozen")]

    # allowed_unit_ids only ever contains a user's own unit and its
    # descendants (see scope_resolver.resolve_user_scope) - it never
    # includes ancestors. But the tree below is built top-down, starting
    # from self-parented Administration roots: a Section or Department
    # admin's own ancestor chain is therefore missing from `units`, so
    # Step 1 never finds a root to walk down from and the whole hierarchy
    # comes back empty for them. Pull their ancestor chain in here, purely
    # so the tree can be constructed and rendered - this does NOT widen
    # what data they can query, since actual access control is enforced
    # separately (via current_user.allowed_unit_ids) in the stats endpoints.
    allowed_unit_ids = set(current_user.allowed_unit_ids)
    if len(current_user.scopes) == 1:
        ancestor_info = admin_units.get_unit_hierarchy(current_user.scopes[0].org_unit_id)
        if ancestor_info:
            if ancestor_info.get("parent_id"):
                allowed_unit_ids.add(ancestor_info["parent_id"])
            if ancestor_info.get("grandparent_id"):
                allowed_unit_ids.add(ancestor_info["grandparent_id"])
    units = [u for u in units if u["UniqueID"] in allowed_unit_ids]

    Administration = []
    Department = defaultdict(list)
    Section = defaultdict(list)

    # -----------------------------
    # Step 1: Administrations
    # Rule: Type == ORG_TYPE_ADMINISTRATION (323).
    # NOTE: this used to detect roots structurally via ParentID == UniqueID
    # (self-parented), which matched old seed/fixture data. Real migrated
    # data from the old HCAT system has ParentID = NULL on Administration
    # rows instead of self-parenting, which left this step (and everything
    # cascading from it) empty. Type is reliably set on real data, so detect
    # roots by type rather than inferring root-ness from ParentID shape.
    # -----------------------------
    for u in units:
        if u["Type"] == ORG_TYPE_ADMINISTRATION:
            # optional name hygiene
            if u["Name"] and u["Name"].strip().upper() != "NULL":
                Administration.append(_unit_payload(u))

    # -----------------------------
    # Step 2: Departments
    # Rule: ParentID == admin_id AND UniqueID != ParentID
    # -----------------------------
    for admin in Administration:
        admin_id = admin["id"]
        Department[admin_id] = []

        for u in units:
            if u["ParentID"] == admin_id and u["UniqueID"] != admin_id:
                if u["Name"] and u["Name"].strip().upper() != "NULL":
                    Department[admin_id].append(_unit_payload(u))

    # -----------------------------
    # Step 3: Sections
    # Rule: ParentID == department_id
    # -----------------------------
    for dept_list in Department.values():
        for dept in dept_list:
            dept_id = dept["id"]
            Section[dept_id] = []

            for u in units:
                if u["ParentID"] == dept_id:
                    if u["Name"] and u["Name"].strip().upper() != "NULL":
                        Section[dept_id].append(_unit_payload(u))

    return {
        "Administration": Administration,
        "Department": dict(Department),
        "Section": dict(Section),
    }

# =========================================================
# REMOVED: Old scope resolution and tree traversal logic
# Phase 2.5: Scoping is now handled by the central scope engine
# All scope computation happens in scope_resolver.py
# Dashboard only uses current_user.allowed_unit_ids
# =========================================================

def _fetch_incidents_in_scope(unit_ids, start_date, end_date):
    """
    Fetch incidents filtered by organizational scope and date range.

    Scoped by TargetOrgUnitID (who the complaint is against / must respond),
    matching the RBAC scoping table_view_service uses — a section/department/
    administration viewer sees cases that concern their unit, not just cases
    their unit happened to file. Uses database-level filtering for performance
    (no in-memory filtering).
    """
    return incident_case.list_incident_cases_filtered_by_target(
        unit_ids=unit_ids,
        start_date=start_date,
        end_date=end_date
    )


def _count_force_closed_subcases(unit_ids: list[int], start_date, end_date) -> int:
    """
    Count subcases with Status = 'ForceClosed' from APP_AdministrativeSubcase.
    
    Filters by:
    - TargetOrgUnitID in unit_ids (organization scope)
    - CreatedAt date range from the linked incident case
    
    Returns:
        Count of force-closed subcases
    """
    from core.database import get_connection
    
    if not unit_ids:
        return 0
    
    conn = get_connection()
    cursor = conn.cursor()
    
    placeholders = ','.join('?' * len(unit_ids))
    
    query = f"""
        SELECT COUNT(*)
        FROM dbo.APP_AdministrativeSubcase s
        INNER JOIN dbo.APP_IncidentCase ic ON s.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE s.Status = 'FORCE_CLOSED'
          AND s.TargetOrgUnitID IN ({placeholders})
          AND CAST(ic.CreatedAt AS DATE) >= ?
          AND CAST(ic.CreatedAt AS DATE) <= ?
    """
    
    params = list(unit_ids) + [start_date, end_date]
    cursor.execute(query, params)
    
    result = cursor.fetchone()[0]
    conn.close()
    
    return result or 0


def _compute_metrics(incidents, force_closed_subcase_count: int = 0):
    metrics = {
        "totalIncidents": len(incidents),
        "uniquePatients": len(
            {i["PatientName"] for i in incidents if i.get("PatientName")}
        ),
        "noticeCount": sum(1 for i in incidents if i.get("RecordTypeID") == 2),
        "openClosed": {
            "open": 0,
            "closed": 0,
            "forciblyClosed": force_closed_subcase_count,  # Count from APP_AdministrativeSubcase
        },
        "severityBreakdown": {
            "high": 0,
            "medium": 0,
            "low": 0,
        },
        "domainBreakdown": {
            "clinical": 0,
            "management": 0,
            "relational": 0,
        },
        "redFlags": 0,
        "neverEvents": 0,
        "ordinary": 0,
    }

    for i in incidents:
        # -------------------------
        # Case status
        # -------------------------
        status_id = i.get("CaseStatusID")

        # forciblyClosed is now counted separately from APP_AdministrativeSubcase
        # Only count open/closed status from incidents
        if status_id in (1,):  # example OPEN
            metrics["openClosed"]["open"] += 1
        elif status_id in (2,):  # example CLOSED
            metrics["openClosed"]["closed"] += 1

        # -------------------------
        # Severity
        # -------------------------
        severity_id = i.get("SeverityID")

        if severity_id == 3:
            metrics["severityBreakdown"]["high"] += 1
        elif severity_id == 2:
            metrics["severityBreakdown"]["medium"] += 1
        elif severity_id == 1:
            metrics["severityBreakdown"]["low"] += 1

        # -------------------------
        # Domain
        # -------------------------
        domain_id = i.get("DomainID")

        if domain_id == 1:
            metrics["domainBreakdown"]["clinical"] += 1
        elif domain_id == 2:
            metrics["domainBreakdown"]["management"] += 1
        elif domain_id == 3:
            metrics["domainBreakdown"]["relational"] += 1

        # -------------------------
        # Clinical risk type: Red Flag / Never Event / Ordinary
        # -------------------------
        clinical_risk_type_id = i.get("ClinicalRiskTypeID")
        if clinical_risk_type_id == 2:
            metrics["redFlags"] += 1
        elif clinical_risk_type_id == 3:
            metrics["neverEvents"] += 1
        else:
            metrics["ordinary"] += 1

    return metrics



def _build_charts(incidents, include_issuing_dept, classification_chart_type="bar", stage_chart_type="bar", department_chart_type="bar"):
    # Build lookup maps (ID -> Name)
    stages = lookups.get_case_stages()
    stage_map = {s["StageID"]: s["StageName"] for s in stages}
    
    classifications = lookups.get_classifications()
    classification_map = {c["ClassificationID"]: c["Classification_AR"] for c in classifications}
    
    org_units = admin_units.get_active_admin_units()
    dept_map = {u["UniqueID"]: u["Name"] for u in org_units}

    severities = lookups.get_severities()
    severity_map = {s["SeverityID"]: s["SeverityName"] for s in severities}

    harm_levels = lookups.get_harm_levels()
    harm_map = {h["HarmID"]: h["HarmLevel"] for h in harm_levels}

    categories = lookups.get_categories()
    category_map = {c["CategoryID"]: c["CategoryName"] for c in categories}

    subcategories = lookups.get_subcategories()
    subcategory_map = {s["SubCategoryID"]: s["SubCategoryName"] for s in subcategories}

    # Build classification chart
    classification_data = _top5_with_names(incidents, "ClassificationID", classification_map)

    # Build stage chart
    stage_data = _histogram_with_names(incidents, "StageID", stage_map)

    # Build severity / harm / category / subcategory distribution charts (Session 5)
    severity_data = _histogram_with_names(incidents, "SeverityID", severity_map)
    harm_data = _histogram_with_names(incidents, "HarmLevelID", harm_map)
    category_data = _histogram_with_names(incidents, "CategoryID", category_map)
    subcategory_data = _top_n_with_other(incidents, "SubCategoryID", subcategory_map, n=8)

    # Build department chart (top 8 by volume, remainder grouped as "Other")
    department_data = None
    if include_issuing_dept:
        department_data = _top_n_with_other(incidents, "IssuingOrgUnitID", dept_map, n=8)

    charts = {
        "classification": {
            "type": classification_chart_type,
            "data": classification_data
        },
        "stage": {
            "type": stage_chart_type,
            "data": stage_data
        },
        "severity": {
            "type": "bar",
            "data": severity_data
        },
        "harm": {
            "type": "bar",
            "data": harm_data
        },
        "category": {
            "type": "bar",
            "data": category_data
        },
        "subcategory": {
            "type": "bar",
            "data": subcategory_data
        },
    }

    if department_data:
        charts["department"] = {
            "type": department_chart_type,
            "data": department_data
        }

    return charts


def _build_recent_activity(incidents, limit=7):
    incidents = sorted(incidents, key=lambda x: x["CreatedAt"], reverse=True)
    return [
        {
            "timestamp": i["CreatedAt"].isoformat(),
            "description": i["ComplaintText"][:120],
            "severity": i["SeverityID"],
            "status": i["CaseStatusID"],
        }
        for i in incidents[:limit]
    ]


def _compute_trends(previous, current):
    def pct(prev, curr):
        if prev == 0:
            return 0
        return int(((curr - prev) / prev) * 100)

    def direction(val):
        return "up" if val >= 0 else "down"

    value = pct(previous["totalIncidents"], current["totalIncidents"])
    return {
        "incidentsPatients": {"value": abs(value), "direction": direction(value)}
    }


# =========================================================
# SMALL UTILS
# =========================================================

def _top5(items, key):
    counter = Counter(i[key] for i in items)
    return [
        {"classification": k, "count": v}
        for k, v in counter.most_common(5)
    ]


def _histogram(items, key):
    counter = Counter(i[key] for i in items)
    return [{"stage": k, "count": v} for k, v in counter.items()]


def _top5_with_names(items, key, id_name_map):
    """
    Get top 5 items by count, with names instead of IDs.
    id_name_map: dict mapping ID to name
    """
    counter = Counter(i[key] for i in items if i.get(key))
    result = []
    for id_val, count in counter.most_common(5):
        name = id_name_map.get(id_val, f"Unknown ({id_val})")
        result.append({"name": name, "count": count})
    return result


def _histogram_with_names(items, key, id_name_map):
    """
    Get histogram with names instead of IDs.
    id_name_map: dict mapping ID to name
    """
    counter = Counter(i[key] for i in items if i.get(key))
    result = []
    for id_val, count in counter.items():
        name = id_name_map.get(id_val, f"Unknown ({id_val})")
        result.append({"name": name, "count": count})
    return result


def _top_n_with_other(items, key, id_name_map, n=8):
    """
    Return the top-N entries by count, grouping the remainder into a single
    'أخرى (Other)' entry.  If there are n or fewer unique values the 'Other'
    bucket is omitted entirely.
    """
    counter = Counter(i[key] for i in items if i.get(key))
    top = counter.most_common(n)
    result = [
        {"name": id_name_map.get(id_val, f"Unknown ({id_val})"), "count": count}
        for id_val, count in top
    ]
    other_count = sum(count for _, count in counter.most_common()[n:])
    if other_count > 0:
        result.append({"name": "أخرى (Other)", "count": other_count})
    return result


def _unit_payload(u):
    return {
        "id": u["UniqueID"],
        "nameAr": u["Name"],
        "nameEn": u["Name"],
    }
