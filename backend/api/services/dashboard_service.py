from datetime import date, timedelta
from collections import Counter, defaultdict
from ..db_layer import incident_case, admin_units,lookups

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
    """

    scope_unit_ids, include_issuing_dept = _resolve_scope(
        scope,
        administration_id,
        department_id,
        section_id,
    )

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


def get_dashboard_hierarchy() -> dict:
    """
    Returns organizational hierarchy for dashboard selectors
    based on the REAL AdminsrationUnit structure.

    Hierarchy rules (as implemented in DB):
    - Administration: ParentID == UniqueID
    - Department: ParentID == Administration.UniqueID AND UniqueID != ParentID
    - Section: ParentID == Department.UniqueID
    """

    raw_units = admin_units.get_admin_unit_tree()
    units = [_row_to_dict(u) for u in raw_units]

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

def _resolve_scope(scope, admin_id, dept_id, section_id):
    raw_units = admin_units.get_admin_unit_tree()
    units = [_row_to_dict(u) for u in raw_units]

    if scope == "hospital":
        return [u["UniqueID"] for u in units], True

    if scope == "administration":
        if admin_id is None:
            raise ValueError("administration_id cannot be None for administration scope")
        descendants = _collect_descendants(units, admin_id)
        if not descendants:
            raise ValueError(f"No units found for administration_id: {admin_id}")
        return descendants, True

    if scope == "department":
        if dept_id is None:
            raise ValueError("department_id cannot be None for department scope")
        descendants = _collect_descendants(units, dept_id)
        if not descendants:
            raise ValueError(f"No units found for department_id: {dept_id}")
        return descendants, True

    if scope == "section":
        if section_id is None:
            raise ValueError("section_id cannot be None for section scope")
        return [section_id], False

    raise ValueError("Invalid scope")



def _collect_descendants(units, root_id):
    """
    Collect root_id and all its descendants using iterative traversal.
    """
    result = set()
    stack = [root_id]
    visited = set()

    while stack:
        current = stack.pop()
        
        # Avoid infinite loops
        if current in visited:
            continue
        
        visited.add(current)
        result.add(current)
        
        # Find all children where ParentID == current AND UniqueID != current
        for u in units:
            child_id = u["UniqueID"]
            parent_id = u["ParentID"]
            
            # A child is where ParentID matches current and it's not the same as current
            if parent_id == current and child_id != current and child_id not in visited:
                stack.append(child_id)
    
    return list(result)


def _fetch_incidents_in_scope(unit_ids, start_date, end_date):
    all_incidents = incident_case.list_incident_cases()
    return [
        i for i in all_incidents
        if i["IssuingOrgUnitID"] in unit_ids
        and start_date <= i["CreatedAt"].date() <= end_date
    ]


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
