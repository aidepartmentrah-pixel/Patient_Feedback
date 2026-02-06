from datetime import date, timedelta
from collections import Counter, defaultdict
from ..db_layer import incident_case, admin_units, lookups
from ..schemas.auth_models import CurrentUser
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

    metrics = _compute_metrics(incidents)
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

    trends = _compute_trends(
        _compute_metrics(previous_incidents),
        metrics,
    )

    return {
        "metrics": metrics,
        "trends": trends,
        "charts": charts,
        "recentActivity": recent_activity,
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
    
    # Filter to only units in user's allowed scope
    allowed_unit_ids = current_user.allowed_unit_ids
    units = [u for u in units if u["UniqueID"] in allowed_unit_ids]

    Administration = []
    Department = defaultdict(list)
    Section = defaultdict(list)

    # -----------------------------
    # Step 1: Administrations
    # Rule: ParentID == UniqueID
    # -----------------------------
    for u in units:
        if u["ParentID"] == u["UniqueID"]:
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
    
    Uses database-level filtering for performance (no in-memory filtering).
    """
    return incident_case.list_incident_cases_filtered(
        unit_ids=unit_ids,
        start_date=start_date,
        end_date=end_date
    )


def _compute_metrics(incidents):
    metrics = {
        "totalIncidents": len(incidents),
        "uniquePatients": len(
            {i["PatientName"] for i in incidents if i.get("PatientName")}
        ),
        "openClosed": {
            "open": 0,
            "closed": 0,
            "forciblyClosed": 0,
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
    }

    for i in incidents:
        # -------------------------
        # Case status & Explanation Status
        # -------------------------
        status_id = i.get("CaseStatusID")
        explanation_status_id = i.get("ExplanationStatusID")

        # Check for Forcibly Closed (ExplanationStatusID == 3)
        if explanation_status_id == 3:
            metrics["openClosed"]["forciblyClosed"] += 1
        elif status_id in (1,):  # example OPEN
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
        # Red flags
        # -------------------------
        clinical_risk_type_id = i.get("ClinicalRiskTypeID")
        if clinical_risk_type_id == 2:
            metrics["redFlags"] += 1

    return metrics



def _build_charts(incidents, include_issuing_dept, classification_chart_type="bar", stage_chart_type="bar", department_chart_type="bar"):
    # Build lookup maps (ID -> Name)
    stages = lookups.get_case_stages()
    stage_map = {s["StageID"]: s["StageName"] for s in stages}
    
    classifications = lookups.get_classifications()
    classification_map = {c["ClassificationID"]: c["Classification_AR"] for c in classifications}
    
    org_units = admin_units.get_active_admin_units()
    dept_map = {u["UniqueID"]: u["Name"] for u in org_units}
    
    # Build classification chart
    classification_data = _top5_with_names(incidents, "ClassificationID", classification_map)
    
    # Build stage chart
    stage_data = _histogram_with_names(incidents, "StageID", stage_map)
    
    # Build department chart
    department_data = None
    if include_issuing_dept:
        department_data = _histogram_with_names(incidents, "IssuingOrgUnitID", dept_map)
    
    charts = {
        "classification": {
            "type": classification_chart_type,
            "data": classification_data
        },
        "stage": {
            "type": stage_chart_type,
            "data": stage_data
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


def _unit_payload(u):
    return {
        "id": u["UniqueID"],
        "nameAr": u["Name"],
        "nameEn": u["Name"],
    }
