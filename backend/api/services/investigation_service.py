from datetime import date, datetime
from collections import defaultdict
from typing import Literal, Optional
from core.database import get_connection
from ..db_layer import admin_units, lookups


# =========================================================
# TYPE DEFINITIONS
# =========================================================

TreeType = Literal[
    "incident_count",
    "domain_distribution_numbers",
    "domain_distribution_percentage",
    "severity_distribution_numbers",
    "severity_distribution_percentage",
    "red_flag_incidents",
    "never_event_incidents",
    "notice_count",
]


# =========================================================
# CONSTANTS
# =========================================================

# Red Flag and Never Event are determined by ClinicalRiskTypeID:
# - ClinicalRiskTypeID = 2 -> Red Flag
# - ClinicalRiskTypeID = 3 -> Never Event
RED_FLAG_CLINICAL_RISK_TYPE = 2
NEVER_EVENT_CLINICAL_RISK_TYPE = 3

# APP_IncidentCase.RecordTypeID values:
# - 1 -> Complaint (default)
# - 2 -> Notice (positive feedback / recognition)
RECORD_TYPE_COMPLAINT = 1
RECORD_TYPE_NOTICE = 2


# =========================================================
# PUBLIC SERVICE FUNCTIONS
# =========================================================

def get_investigation_tree(
    *,
    start_date: date,
    end_date: date,
    tree_type: TreeType,
    administration_id: int | None = None,
    department_id: int | None = None,
    section_id: int | None = None,
) -> dict:
    """
    Get hierarchical investigation tree with aggregated incident data.

    Args:
        start_date: Start of the investigation period
        end_date: End of the investigation period
        tree_type: Type of aggregation/visualization
        administration_id: Filter to specific administration
        department_id: Filter to specific department
        section_id: Filter to specific section

    Returns:
        Dictionary with tree structure and metadata
    """
    print(f"Investigation period: {start_date} to {end_date}")

    # -------------------------
    # Determine scope level
    # -------------------------
    scope_level = "hospital"
    scope_unit_id = None
    
    if section_id is not None:
        scope_level = "section"
        scope_unit_id = section_id
    elif department_id is not None:
        scope_level = "department"
        scope_unit_id = department_id
    elif administration_id is not None:
        scope_level = "administration"
        scope_unit_id = administration_id
    
    # -------------------------
    # Fetch organizational hierarchy
    # -------------------------
    org_hierarchy = _build_org_hierarchy(scope_unit_id, scope_level)
    
    print(f"Organizational hierarchy built: {len(org_hierarchy['root_nodes'])} root nodes")
    
    # -------------------------
    # Determine record type filter
    # notice_count uses notices only; all other tree types use complaints only
    # -------------------------
    record_type_id = RECORD_TYPE_NOTICE if tree_type == "notice_count" else RECORD_TYPE_COMPLAINT

    # -------------------------
    # Fetch incidents for the season and scope
    # -------------------------
    incidents = _fetch_incidents_for_season(
        start_date,
        end_date,
        scope_unit_id,
        scope_level,
        record_type_id=record_type_id,
    )
    
    print(f"Fetched {len(incidents)} incidents for the season")
    
    # -------------------------
    # Build tree based on tree_type
    # -------------------------
    # Handle case where no organizational structure found
    if not org_hierarchy["root_nodes"]:
        raise ValueError(f"No organizational units found for the specified scope")
    
    if tree_type == "incident_count":
        tree = _build_incident_count_tree(org_hierarchy, incidents)
        summary = _build_incident_count_summary(incidents, org_hierarchy)
    
    elif tree_type == "domain_distribution_numbers":
        tree = _build_domain_distribution_tree(org_hierarchy, incidents, as_percentage=False)
        summary = _build_domain_distribution_summary(incidents, as_percentage=False)
    
    elif tree_type == "domain_distribution_percentage":
        tree = _build_domain_distribution_tree(org_hierarchy, incidents, as_percentage=True)
        summary = _build_domain_distribution_summary(incidents, as_percentage=True)
    
    elif tree_type == "severity_distribution_numbers":
        tree = _build_severity_distribution_tree(org_hierarchy, incidents, as_percentage=False)
        summary = _build_severity_distribution_summary(incidents, as_percentage=False)
    
    elif tree_type == "severity_distribution_percentage":
        tree = _build_severity_distribution_tree(org_hierarchy, incidents, as_percentage=True)
        summary = _build_severity_distribution_summary(incidents, as_percentage=True)
    
    elif tree_type == "red_flag_incidents":
        tree = _build_red_flag_tree(org_hierarchy, incidents)
        summary = _build_red_flag_summary(incidents)
    
    elif tree_type == "never_event_incidents":
        tree = _build_never_event_tree(org_hierarchy, incidents)
        summary = _build_never_event_summary(incidents)

    elif tree_type == "notice_count":
        tree = _build_incident_count_tree(org_hierarchy, incidents)
        summary = _build_incident_count_summary(incidents, org_hierarchy)

    else:
        raise ValueError(f"Invalid tree_type: {tree_type}")
    
    # -------------------------
    # Build response
    # -------------------------
    return {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "tree_type": tree_type,
        "scope": {
            "level": scope_level,
            "administration_id": administration_id,
            "department_id": department_id,
            "section_id": section_id,
        },
        "tree": tree,
        "summary": summary,
    }


def get_available_seasons() -> dict:
    """
    Get list of available investigation seasons/periods.
    
    Auto-generates missing seasons to ensure autonomous operation.
    Uses date-based detection for current season (not IsDone flag).
    
    Returns:
        Dictionary with seasons array and current season
    """
    from datetime import date as dt_date
    
    conn = get_connection()
    cursor = conn.cursor()
    
    today = dt_date.today()
    
    # =============================================
    # AUTO-GENERATE MISSING SEASONS
    # =============================================
    QUARTER_DATE_RANGES = {
        1: {"start_month": 1, "start_day": 1, "end_month": 3, "end_day": 31},
        2: {"start_month": 4, "start_day": 1, "end_month": 6, "end_day": 30},
        3: {"start_month": 7, "start_day": 1, "end_month": 9, "end_day": 30},
        4: {"start_month": 10, "start_day": 1, "end_month": 12, "end_day": 31},
    }
    
    current_year = today.year
    years_ahead = 2
    
    # Get the current max UniqueID
    cursor.execute("SELECT ISNULL(MAX(UniqueID), 0) FROM dbo.Season")
    next_id = cursor.fetchone()[0] + 1
    
    for year in range(current_year, current_year + years_ahead + 1):
        for quarter in range(1, 5):
            season_name = f"Q{quarter}-{year}"
            
            # Check if exists
            cursor.execute(
                "SELECT UniqueID FROM dbo.Season WHERE SeasonName = ?",
                (season_name,)
            )
            
            if cursor.fetchone() is not None:
                continue
            
            # Create it
            q = QUARTER_DATE_RANGES[quarter]
            start_date = dt_date(year, q["start_month"], q["start_day"])
            end_date = dt_date(year, q["end_month"], q["end_day"])
            
            cursor.execute(
                """
                INSERT INTO dbo.Season (UniqueID, SeasonName, StartDate, EndDate, IsDone, Frozen)
                VALUES (?, ?, ?, ?, 0, 0)
                """,
                (next_id, season_name, start_date, end_date)
            )
            next_id += 1
            print(f"[Investigation] Auto-created season: {season_name}")
    
    conn.commit()
    
    # =============================================
    # FETCH ALL SEASONS
    # =============================================
    cursor.execute(
        """
        SELECT 
            UniqueID,
            SeasonName,
            StartDate,
            EndDate,
            IsDone
        FROM dbo.Season
        WHERE Frozen = 0
        ORDER BY StartDate DESC
        """
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    current_season_id = None
    seasons = []
    
    for row in rows:
        season_id = str(row.UniqueID)
        season_name = row.SeasonName
        start_date = row.StartDate
        end_date = row.EndDate
        is_done = bool(row.IsDone)
        
        # Determine current season by DATE RANGE (correct approach)
        is_current = False
        if start_date and end_date:
            if start_date <= today <= end_date:
                is_current = True
                current_season_id = season_id
        
        seasons.append({
            "season_id": season_id,
            "season_label": season_name,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "is_current": is_current,
        })
    
    return {
        "seasons": seasons,
        "current_season": current_season_id,
    }


def get_organizational_hierarchy() -> dict:
    """
    Get organizational hierarchy for cascading selectors.
    Reuses admin_units structure.
    
    Returns:
        Dictionary with administrations, departments, sections arrays
    """
    raw_units = admin_units.get_admin_unit_tree()
    
    administrations = []
    departments = []
    sections = []
    
    for unit in raw_units:
        unit_id = unit.UniqueID
        parent_id = unit.ParentID
        name = unit.Name
        unit_type = unit.Type
        
        # Administration: ParentID == UniqueID
        if parent_id == unit_id:
            administrations.append({
                "id": unit_id,
                "name_en": name,
                "name_ar": name,  # Assuming name is bilingual or Arabic
            })
        
        # Department: ParentID points to administration, UniqueID != ParentID
        elif parent_id != unit_id:
            # Check if parent is an administration
            parent_is_admin = False
            for u in raw_units:
                if u.UniqueID == parent_id and u.ParentID == u.UniqueID:
                    parent_is_admin = True
                    break
            
            if parent_is_admin:
                departments.append({
                    "id": unit_id,
                    "administration_id": parent_id,
                    "name_en": name,
                    "name_ar": name,
                })
            else:
                # Section: ParentID points to department
                sections.append({
                    "id": unit_id,
                    "department_id": parent_id,
                    "name_en": name,
                    "name_ar": name,
                })
    
    return {
        "administrations": administrations,
        "departments": departments,
        "sections": sections,
    }


# =========================================================
# PRIVATE HELPER FUNCTIONS - DATA FETCHING
# =========================================================

def _get_season_info(season: str) -> dict | None:
    """
    Get season information by ID or name.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Try to parse as integer ID
    try:
        season_id = int(season)
        cursor.execute(
            """
            SELECT UniqueID, SeasonName, StartDate, EndDate
            FROM dbo.Season
            WHERE UniqueID = ? AND Frozen = 0
            """,
            season_id
        )
    except ValueError:
        # Try to match by name
        cursor.execute(
            """
            SELECT UniqueID, SeasonName, StartDate, EndDate
            FROM dbo.Season
            WHERE SeasonName = ? AND Frozen = 0
            """,
            season
        )
    
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "season_id": str(row.UniqueID),
        "season_label": row.SeasonName,
        "start_date": row.StartDate,
        "end_date": row.EndDate,
    }


def _fetch_incidents_for_season(
    start_date: date,
    end_date: date,
    scope_unit_id: int | None,
    scope_level: str,
    record_type_id: int = RECORD_TYPE_COMPLAINT,
) -> list[dict]:
    """
    Fetch incidents for a date range, filtered by record type and optional org scope.

    record_type_id:
        RECORD_TYPE_COMPLAINT (1) — complaints only (default for all complaint trees)
        RECORD_TYPE_NOTICE    (2) — notices only
    """
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    SELECT
        ic.IncidentRequestCaseID,
        td.DepartmentID AS TargetOrgUnitID,
        ic.DomainID,
        ic.SeverityID,
        ic.CategoryID,
        ic.HarmLevelID,
        ic.ClinicalRiskTypeID,
        ic.FeedbackRecievedDate
    FROM dbo.APP_IncidentCase ic
    INNER JOIN dbo.APP_IncidentCaseTargetDepartment td
        ON ic.IncidentRequestCaseID = td.IncidentRequestCaseID
        AND td.IsPrimary = 1
    WHERE ic.FeedbackRecievedDate >= ?
      AND ic.FeedbackRecievedDate <= ?
      AND ic.RecordTypeID = ?
    """

    params = [start_date, end_date, record_type_id]

    # If scope is specified, filter by target organizational unit and its descendants
    if scope_unit_id is not None:
        # Get all descendant unit IDs
        descendant_ids = _get_descendant_unit_ids(scope_unit_id)

        print(f"Filtering by scope unit {scope_unit_id}, found {len(descendant_ids)} descendant units: {descendant_ids}")

        if descendant_ids:
            placeholders = ','.join('?' * len(descendant_ids))
            query += f" AND td.DepartmentID IN ({placeholders})"
            params.extend(descendant_ids)
        else:
            # No descendants found, return empty list
            print(f"Warning: No descendant units found for scope_unit_id={scope_unit_id}")
            conn.close()
            return []
    
    cursor.execute(query, params)
    
    rows = cursor.fetchall()
    columns = [col[0] for col in cursor.description]
    
    conn.close()
    
    return [dict(zip(columns, row)) for row in rows]


def _get_descendant_unit_ids(unit_id: int) -> list[int]:
    """
    Get all descendant unit IDs for a given unit (including the unit itself).
    Uses SQL Server recursive CTE syntax.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Use SQL Server recursive CTE to get all descendants
    cursor.execute(
        """
        WITH UnitHierarchy AS (
            -- Anchor: Start with the given unit
            SELECT UniqueID, ParentID
            FROM dbo.AdminsrationUnit
            WHERE UniqueID = ?
            
            UNION ALL
            
            -- Recursive: Get all children
            SELECT u.UniqueID, u.ParentID
            FROM dbo.AdminsrationUnit u
            INNER JOIN UnitHierarchy h ON u.ParentID = h.UniqueID
            WHERE u.UniqueID != u.ParentID  -- Avoid infinite loop on administrations
        )
        SELECT UniqueID FROM UnitHierarchy
        """,
        unit_id
    )
    
    rows = cursor.fetchall()
    conn.close()
    
    return [row.UniqueID for row in rows]


def _build_org_hierarchy(scope_unit_id: int | None, scope_level: str) -> dict:
    """
    Build organizational hierarchy tree structure.
    Returns nested dictionary with unit info and children.
    """
    raw_units = admin_units.get_admin_unit_tree()
    
    # Build lookup dictionaries
    units_by_id = {}
    units_by_parent = defaultdict(list)
    
    for unit in raw_units:
        unit_id = unit.UniqueID
        parent_id = unit.ParentID
        
        unit_info = {
            "node_id": unit_id,
            "node_name": unit.Name,
            "node_name_ar": unit.Name,
            "parent_id": parent_id if parent_id != unit_id else None,
            "children": [],
        }
        
        # Determine node type
        if parent_id == unit_id:
            unit_info["node_type"] = "administration"
            unit_info["level"] = 0
        else:
            # Check if parent is administration
            parent_is_admin = False
            for u in raw_units:
                if u.UniqueID == parent_id and u.ParentID == u.UniqueID:
                    parent_is_admin = True
                    break
            
            if parent_is_admin:
                unit_info["node_type"] = "department"
                unit_info["level"] = 1
            else:
                unit_info["node_type"] = "section"
                unit_info["level"] = 2
        
        units_by_id[unit_id] = unit_info
        if parent_id != unit_id:  # Don't add self-references to parent lookup
            units_by_parent[parent_id].append(unit_info)
    
    # Build tree structure recursively
    def build_tree(unit_id):
        unit = units_by_id.get(unit_id)
        if not unit:
            return None
        
        # Make a copy to avoid modifying the original
        unit_copy = unit.copy()
        unit_copy["children"] = []
        
        # Add children recursively
        for child_unit in units_by_parent.get(unit_id, []):
            child_tree = build_tree(child_unit["node_id"])
            if child_tree:
                unit_copy["children"].append(child_tree)
        
        return unit_copy
    
    # If scope is specified, start from that unit
    root_nodes = []
    if scope_unit_id is not None:
        root_tree = build_tree(scope_unit_id)
        if root_tree:
            root_nodes = [root_tree]
    else:
        # Get all top-level administrations
        for unit in raw_units:
            if unit.ParentID == unit.UniqueID:
                root_tree = build_tree(unit.UniqueID)
                if root_tree:
                    root_nodes.append(root_tree)
    
    return {
        "units_by_id": units_by_id,
        "root_nodes": root_nodes,
    }


# =========================================================
# PRIVATE HELPER FUNCTIONS - TREE BUILDING
# =========================================================

def _build_incident_count_tree(org_hierarchy: dict, incidents: list[dict]) -> list[dict]:
    """
    Build tree with incident counts for each node.
    """
    # Count incidents per target unit
    incident_counts = defaultdict(int)
    for incident in incidents:
        unit_id = incident["TargetOrgUnitID"]
        if unit_id:
            incident_counts[unit_id] += 1
    
    # Populate tree nodes with counts
    def populate_node(node):
        unit_id = node["node_id"]
        
        # Get direct count
        direct_count = incident_counts[unit_id]
        
        # Get children counts (recursive)
        child_count = 0
        for child in node["children"]:
            populate_node(child)
            child_count += child["value"]
        
        # Total count = direct + children
        node["value"] = direct_count + child_count
        
        return node
    
    tree = []
    for root in org_hierarchy["root_nodes"]:
        tree.append(populate_node(root.copy()))
    
    return tree


def _build_domain_distribution_tree(
    org_hierarchy: dict,
    incidents: list[dict],
    as_percentage: bool
) -> list[dict]:
    """
    Build tree with domain distribution for each node.
    """
    # Get domain lookup
    domains = {d["DomainID"]: d for d in lookups.get_domains()}
    
    # Count incidents by target unit and domain
    unit_domain_counts = defaultdict(lambda: defaultdict(int))
    for incident in incidents:
        unit_id = incident["TargetOrgUnitID"]
        domain_id = incident["DomainID"]
        if unit_id and domain_id:
            domain_name = domains.get(domain_id, {}).get("DomainName", f"Domain_{domain_id}")
            unit_domain_counts[unit_id][domain_name] += 1
    
    # Populate tree nodes
    def populate_node(node):
        unit_id = node["node_id"]
        
        # Aggregate domain counts from this node and children
        domain_totals = defaultdict(int)
        
        # Add direct counts
        for domain_name, count in unit_domain_counts[unit_id].items():
            domain_totals[domain_name] += count
        
        # Add children counts (recursive)
        for child in node["children"]:
            populate_node(child)
            for domain_name, count in child.get("domain_breakdown", {}).items():
                domain_totals[domain_name] += count
        
        total_incidents = sum(domain_totals.values())
        
        # Convert to percentage if needed
        if as_percentage:
            if total_incidents > 0:
                domain_breakdown = {
                    domain: round((count / total_incidents) * 100, 1)
                    for domain, count in domain_totals.items()
                }
            else:
                domain_breakdown = {}
            node["value"] = 100.0 if total_incidents > 0 else 0.0
        else:
            domain_breakdown = dict(domain_totals)
            node["value"] = total_incidents
        
        node["domain_breakdown"] = domain_breakdown
        node["total_incidents"] = total_incidents
        
        return node
    
    tree = []
    for root in org_hierarchy["root_nodes"]:
        tree.append(populate_node(root.copy()))
    
    return tree


def _build_severity_distribution_tree(
    org_hierarchy: dict,
    incidents: list[dict],
    as_percentage: bool
) -> list[dict]:
    """
    Build tree with severity distribution for each node.
    """
    # Get severity lookup (simplified - map severity IDs to high/medium/low)
    # Adjust based on your actual severity definitions
    severity_mapping = {
        1: "low",
        2: "low",
        3: "medium",
        4: "medium",
        5: "high",
        6: "high",
    }
    
    # Count incidents by target unit and severity
    unit_severity_counts = defaultdict(lambda: defaultdict(int))
    for incident in incidents:
        unit_id = incident["TargetOrgUnitID"]
        severity_id = incident["SeverityID"]
        if unit_id and severity_id:
            severity_level = severity_mapping.get(severity_id, "unknown")
            unit_severity_counts[unit_id][severity_level] += 1
    
    # Populate tree nodes
    def populate_node(node):
        unit_id = node["node_id"]
        
        # Aggregate severity counts
        severity_totals = defaultdict(int)
        
        # Add direct counts
        for severity, count in unit_severity_counts[unit_id].items():
            severity_totals[severity] += count
        
        # Add children counts (recursive)
        for child in node["children"]:
            populate_node(child)
            for severity, count in child.get("severity_breakdown", {}).items():
                severity_totals[severity] += count
        
        total_incidents = sum(severity_totals.values())
        
        # Convert to percentage if needed
        if as_percentage:
            if total_incidents > 0:
                severity_breakdown = {
                    severity: round((count / total_incidents) * 100, 1)
                    for severity, count in severity_totals.items()
                }
            else:
                severity_breakdown = {}
            node["value"] = 100.0 if total_incidents > 0 else 0.0
        else:
            severity_breakdown = dict(severity_totals)
            node["value"] = total_incidents
        
        node["severity_breakdown"] = severity_breakdown
        node["total_incidents"] = total_incidents
        
        return node
    
    tree = []
    for root in org_hierarchy["root_nodes"]:
        tree.append(populate_node(root.copy()))
    
    return tree


def _build_red_flag_tree(org_hierarchy: dict, incidents: list[dict]) -> list[dict]:
    """
    Build tree with red flag incident counts.
    Red flags = ClinicalRiskTypeID = 2
    """
    # Count red flags and total incidents per unit
    unit_red_flags = defaultdict(int)
    unit_totals = defaultdict(int)
    
    for incident in incidents:
        unit_id = incident["TargetOrgUnitID"]
        clinical_risk_type_id = incident.get("ClinicalRiskTypeID")

        if unit_id:
            unit_totals[unit_id] += 1
            if clinical_risk_type_id == RED_FLAG_CLINICAL_RISK_TYPE:
                unit_red_flags[unit_id] += 1
    
    # Populate tree nodes
    def populate_node(node):
        unit_id = node["node_id"]
        
        # Aggregate counts
        red_flag_count = unit_red_flags[unit_id]
        total_count = unit_totals[unit_id]
        
        # Add children counts (recursive)
        for child in node["children"]:
            populate_node(child)
            red_flag_count += child["value"]
            total_count += child["total_incidents"]
        
        # Calculate percentage
        red_flag_percentage = round((red_flag_count / total_count) * 100, 2) if total_count > 0 else 0.0
        
        node["value"] = red_flag_count
        node["total_incidents"] = total_count
        node["red_flag_percentage"] = red_flag_percentage
        
        return node
    
    tree = []
    for root in org_hierarchy["root_nodes"]:
        tree.append(populate_node(root.copy()))
    
    return tree


def _build_never_event_tree(org_hierarchy: dict, incidents: list[dict]) -> list[dict]:
    """
    Build tree with never event incident counts.
    Never events = ClinicalRiskTypeID = 3
    """
    # Count never events and total incidents per unit
    unit_never_events = defaultdict(int)
    unit_totals = defaultdict(int)
    
    for incident in incidents:
        unit_id = incident["TargetOrgUnitID"]
        clinical_risk_type_id = incident.get("ClinicalRiskTypeID")

        if unit_id:
            unit_totals[unit_id] += 1

            if clinical_risk_type_id == NEVER_EVENT_CLINICAL_RISK_TYPE:
                unit_never_events[unit_id] += 1
    
    # Populate tree nodes
    def populate_node(node):
        unit_id = node["node_id"]
        
        # Aggregate counts
        never_event_count = unit_never_events[unit_id]
        total_count = unit_totals[unit_id]
        
        # Add children counts (recursive)
        for child in node["children"]:
            populate_node(child)
            never_event_count += child["value"]
            total_count += child["total_incidents"]
        
        # Calculate percentage
        never_event_percentage = round((never_event_count / total_count) * 100, 2) if total_count > 0 else 0.0
        
        node["value"] = never_event_count
        node["total_incidents"] = total_count
        node["never_event_percentage"] = never_event_percentage
        
        return node
    
    tree = []
    for root in org_hierarchy["root_nodes"]:
        tree.append(populate_node(root.copy()))
    
    return tree


# =========================================================
# PRIVATE HELPER FUNCTIONS - SUMMARY BUILDING
# =========================================================

def _build_incident_count_summary(incidents: list[dict], org_hierarchy: dict) -> dict:
    """
    Build summary for incident count tree.
    """
    # Count nodes by type from the hierarchy
    admin_count = 0
    dept_count = 0
    section_count = 0
    
    def count_nodes(node):
        nonlocal admin_count, dept_count, section_count
        if node["node_type"] == "administration":
            admin_count += 1
        elif node["node_type"] == "department":
            dept_count += 1
        elif node["node_type"] == "section":
            section_count += 1
        
        for child in node.get("children", []):
            count_nodes(child)
    
    for root in org_hierarchy["root_nodes"]:
        count_nodes(root)
    
    return {
        "total_incidents": len(incidents),
        "administration_count": admin_count,
        "department_count": dept_count,
        "section_count": section_count,
    }


def _build_domain_distribution_summary(incidents: list[dict], as_percentage: bool) -> dict:
    """
    Build summary for domain distribution tree.
    """
    domains = {d["DomainID"]: d for d in lookups.get_domains()}
    
    domain_counts = defaultdict(int)
    for incident in incidents:
        domain_id = incident["DomainID"]
        if domain_id:
            domain_name = domains.get(domain_id, {}).get("DomainName", f"Domain_{domain_id}")
            domain_counts[domain_name] += 1
    
    total = sum(domain_counts.values())
    
    if as_percentage and total > 0:
        overall_breakdown = {
            domain: round((count / total) * 100, 1)
            for domain, count in domain_counts.items()
        }
    else:
        overall_breakdown = dict(domain_counts)
    
    return {
        "total_incidents": total,
        "overall_domain_breakdown": overall_breakdown,
    }


def _build_severity_distribution_summary(incidents: list[dict], as_percentage: bool) -> dict:
    """
    Build summary for severity distribution tree.
    """
    severity_mapping = {
        1: "low",
        2: "low",
        3: "medium",
        4: "medium",
        5: "high",
        6: "high",
    }
    
    severity_counts = defaultdict(int)
    for incident in incidents:
        severity_id = incident["SeverityID"]
        if severity_id:
            severity_level = severity_mapping.get(severity_id, "unknown")
            severity_counts[severity_level] += 1
    
    total = sum(severity_counts.values())
    
    if as_percentage and total > 0:
        overall_breakdown = {
            severity: round((count / total) * 100, 1)
            for severity, count in severity_counts.items()
        }
    else:
        overall_breakdown = dict(severity_counts)
    
    return {
        "total_incidents": total,
        "overall_severity_breakdown": overall_breakdown,
    }


def _build_red_flag_summary(incidents: list[dict]) -> dict:
    """
    Build summary for red flag tree.
    Red flags = ClinicalRiskTypeID = 2
    """
    red_flag_count = sum(
        1 for incident in incidents
        if incident.get("ClinicalRiskTypeID") == RED_FLAG_CLINICAL_RISK_TYPE
    )
    
    total = len(incidents)
    percentage = round((red_flag_count / total) * 100, 2) if total > 0 else 0.0
    
    return {
        "total_red_flags": red_flag_count,
        "total_incidents": total,
        "overall_red_flag_percentage": percentage,
    }


def _build_never_event_summary(incidents: list[dict]) -> dict:
    """
    Build summary for never event tree.
    Never events = ClinicalRiskTypeID = 3
    """
    never_event_count = sum(
        1 for incident in incidents
        if incident.get("ClinicalRiskTypeID") == NEVER_EVENT_CLINICAL_RISK_TYPE
    )
    
    total = len(incidents)
    percentage = round((never_event_count / total) * 100, 2) if total > 0 else 0.0
    
    return {
        "total_never_events": never_event_count,
        "total_incidents": total,
        "overall_never_event_percentage": percentage,
    }
