"""
Insight Service Layer (Phase 4B - B-I7)

Service layer for analytics/insight endpoints.
Orchestrates DB layer calls with scope enforcement via allowed_unit_ids.

Architecture:
  Router → Service → DB Layer

Functions read current_user.allowed_unit_ids and pass to DB layer.
No SQL queries here - all SQL in insight_db.py.
"""

from typing import Dict, Any, List
from collections import defaultdict
from core.database import get_connection
from ..db_layer import insight_db, administrative_subcase_db
from backend.api.schemas.auth_models import CurrentUser


def get_kpi_summary(current_user: CurrentUser) -> Dict[str, Any]:
    """
    Get KPI summary for current user's scope.
    
    Returns aggregate metrics:
    - Subcase status distribution
    - Action item counts (total, open, completed, overdue)
    - Stuck case count
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        
    Returns:
        Dictionary with KPI metrics
    """
    conn = get_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get subcase status counts
        status_counts = insight_db.get_subcase_status_counts(conn, allowed_unit_ids)
        
        # Get action item counts
        action_items = insight_db.get_action_item_counts(conn, allowed_unit_ids)
        
        # Compute total subcases from status counts
        total_subcases = sum(item['count'] for item in status_counts)
        
        return {
            "total_subcases": total_subcases,
            "by_status": status_counts,
            "action_items": action_items
        }
    finally:
        conn.close()


def get_distribution(current_user: CurrentUser, dimension: str) -> List[Dict[str, Any]]:
    """
    Get subcase distribution by specified dimension.
    
    Currently supports:
    - dimension="status": Distribution by subcase status
    - dimension="org_unit": Distribution by target org unit
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        dimension: Dimension to group by ("status" or "org_unit")
        
    Returns:
        List of distribution items with counts
        Shape: [{"key": value, "count": int}]
        
    Raises:
        ValueError: If dimension is not supported
    """
    # Validate dimension
    valid_dimensions = ["status", "org_unit"]
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension: '{dimension}'. Must be one of: {valid_dimensions}")
    
    conn = get_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        if dimension == "status":
            # Get status distribution
            raw_results = insight_db.get_subcase_status_counts(conn, allowed_unit_ids)
            # Transform to standard shape: {"key": status_value, "count": count}
            return [
                {"key": item["status"], "count": item["count"]}
                for item in raw_results
            ]
        
        elif dimension == "org_unit":
            # Get org unit distribution
            raw_results = insight_db.get_subcase_org_unit_counts(conn, allowed_unit_ids)
            # Transform to standard shape: {"key": org_unit_id, "count": count}
            return [
                {"key": item["target_org_unit_id"], "count": item["count"]}
                for item in raw_results
            ]
    finally:
        conn.close()


def get_trend(current_user: CurrentUser, bucket: str) -> List[Dict[str, Any]]:
    """
    Get time-series trend of subcase creation.
    
    Buckets:
    - "day": Daily trend (YYYY-MM-DD)
    - "month": Monthly trend (YYYY-MM)
    - "year": Yearly trend (YYYY)
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        bucket: Time bucket granularity ("day", "month", or "year")
        
    Returns:
        List of time buckets with counts, sorted ascending
        Shape: [{"bucket": str, "count": int}]
        
    Raises:
        ValueError: If bucket is not supported
    """
    # Validate bucket
    valid_buckets = ["day", "month", "year"]
    if bucket not in valid_buckets:
        raise ValueError(f"Invalid bucket: '{bucket}'. Must be one of: {valid_buckets}")
    
    conn = get_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get time buckets from DB
        raw_results = insight_db.get_subcase_created_time_buckets(conn, allowed_unit_ids, bucket)
        
        # Transform: rename bucket_label → bucket
        return [
            {"bucket": item["bucket_label"], "count": item["count"]}
            for item in raw_results
        ]
    finally:
        conn.close()


def get_stuck_cases(current_user: CurrentUser, days_threshold: int = 30) -> List[Dict[str, Any]]:
    """
    Get list of subcases stuck in non-terminal status.
    
    Stuck = UpdatedAt older than threshold AND not in terminal statuses.
    
    A subcase is "stuck" if it has been in current status for >= days_threshold days
    and is not in a terminal status (ADMIN_APPROVED, SECTION_DENIED, FORCE_CLOSED).
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        days_threshold: Minimum days in status to be considered "stuck" (default: 30)
        
    Returns:
        List of stuck subcases with subcase_id, status, target_org_unit_id,
        updated_at, days_in_stage
    """
    conn = get_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get stuck subcases from DB (returns rows unchanged)
        return insight_db.get_stuck_subcases(conn, allowed_unit_ids, days_threshold)
    finally:
        conn.close()


def get_user_workload(
    current_user: CurrentUser,
    org_unit_id: int = None,
    role: str = None,
    min_items: int = 1,
    sort_by: str = 'pending_count',
    sort_order: str = 'desc'
) -> List[Dict[str, Any]]:
    """
    Get user workload statistics based on assigned action items.
    
    Returns person-centric workload view showing which users have pending work.
    Only counts action items assigned to users for subcases NOT in terminal states.
    
    Enables proactive follow-up (e.g., "Dr. Smith has 10 pending items, let's call them").
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        org_unit_id: Optional filter by organizational unit ID
        role: Optional filter by user role code
        min_items: Only show users with >= N pending items (default: 1)
        sort_by: Sort by 'pending_count', 'oldest_item', or 'user_name' (default: 'pending_count')
        sort_order: Sort order 'asc' or 'desc' (default: 'desc')
        
    Returns:
        List of user workload dicts:
        [
            {
                "user_id": 456,
                "user_name": "Dr. John Smith",
                "user_role": "SECTION_ADMIN",
                "primary_org_unit": "Cardiology Section",
                "pending_count": 10,
                "oldest_item_days": 15
            },
            ...
        ]
    """
    conn = get_connection()
    try:
        # Convert set to list for DB layer
        allowed_unit_ids = list(current_user.allowed_unit_ids)
        
        # Get user workload from DB
        return insight_db.get_user_workload(
            conn=conn,
            allowed_unit_ids=allowed_unit_ids,
            org_unit_filter=org_unit_id,
            role_filter=role,
            min_items=min_items,
            sort_by=sort_by,
            sort_order=sort_order
        )
    finally:
        conn.close()


# =============================================================================
# GROUPED INBOX (Phase 3 - Analytical View)
# =============================================================================

def get_grouped_inbox_for_admin(current_user: CurrentUser) -> List[Dict[str, Any]]:
    """
    Get subcases grouped by target org unit (section/dept/admin).
    
    Each group shows:
    - Section/department name
    - Supervisor name
    - Count of pending subcases
    - List of subcases with full details (description, severity, waiting time)
    
    Subcases within each group are sorted by waiting_days DESC (longest first).
    Groups are sorted by pending_count DESC (busiest first).
    Empty groups (0 pending) are excluded.
    
    Security:
    - Applies Phase 2.5 scope filtering via allowed_unit_ids
    - Only returns org units user has access to
    
    Args:
        current_user: Authenticated user with allowed_unit_ids
        
    Returns:
        List of group dicts:
        [
          {
            "section_id": int,
            "section_name": str,
            "org_type": str,
            "supervisor_name": str,  # "Unassigned" if none
            "pending_count": int,
            "subcases": [...]
          }
        ]
    """
    # Determine which DB query to use based on user role
    # Default to section if no role (defensive)
    role_code = current_user.scopes[0].role_code if current_user.scopes else None
    
    # Get subcases from appropriate DB query
    if role_code == 'DEPARTMENT_ADMIN':
        raw_subcases = administrative_subcase_db.get_subcases_with_details_for_department()
    elif role_code == 'ADMINISTRATION_ADMIN':
        raw_subcases = administrative_subcase_db.get_subcases_with_details_for_administration()
    else:  # Default to section (SECTION_ADMIN or fallback)
        raw_subcases = administrative_subcase_db.get_subcases_with_details_for_section()
    
    # Apply scope filtering - only show org units user has access to
    filtered_subcases = _apply_scope_filter_to_subcases(raw_subcases, current_user)
    
    # Group subcases by target org unit
    grouped = _group_subcases_by_org_unit(filtered_subcases)
    
    # Build group objects with metadata
    groups = []
    for org_unit_id, subcases in grouped.items():
        group = _build_section_group(org_unit_id, subcases)
        groups.append(group)
    
    # Sort groups by pending_count DESC (busiest first)
    groups.sort(key=lambda g: g['pending_count'], reverse=True)
    
    return groups


def get_force_closed_cases(current_user: CurrentUser, status: str) -> List[Dict[str, Any]]:
    """
    Return force-closed subcases for the Insight page tabs.

    Authorization: COMPLAINT_SUPERVISOR and WORKER only.
    Returns all force-closed subcases (not scope-filtered — supervisors need full visibility).

    Args:
        current_user: Authenticated user
        status: 'FORCE_CLOSED_DRAFT' or 'FORCE_CLOSED_COMPLETE'

    Returns:
        List of subcase dicts with case details and force-close metadata.
    """
    from fastapi import HTTPException

    allowed_roles = {'COMPLAINT_SUPERVISOR', 'WORKER'}
    user_roles = {s.role_code for s in current_user.scopes} if current_user.scopes else set()
    if not user_roles & allowed_roles:
        raise HTTPException(
            status_code=403,
            detail="Only COMPLAINT_SUPERVISOR or WORKER can view force-closed cases."
        )

    valid_statuses = {'FORCE_CLOSED_DRAFT', 'FORCE_CLOSED_COMPLETE'}
    if status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"status must be one of {valid_statuses}")

    return administrative_subcase_db.get_force_closed_subcases(status)


def generate_insight_word_export(
    current_user: CurrentUser,
    search_term: str = None,
) -> bytes:
    """
    Gather current Insight page state and generate a Word document.

    Fetches the same data as the Insight page (grouped inbox + force-closed cases),
    applies an optional search filter, then calls the Word generator.

    Args:
        current_user: Authenticated user
        search_term: Optional text filter (mirrors frontend search box)

    Returns:
        bytes of the generated .docx file
    """
    from .insight_word_generator import generate_insight_word_report

    # Active cases (same data as grouped inbox tab)
    grouped_inbox = get_grouped_inbox_for_admin(current_user)

    # Apply search filter when requested (mirrors frontend filterBySearch logic)
    if search_term and search_term.strip():
        grouped_inbox = _apply_search_filter_to_inbox(grouped_inbox, search_term.strip())

    # Force-closed cases — only for authorized roles; silently empty otherwise
    fc_draft: List[Dict[str, Any]] = []
    fc_complete: List[Dict[str, Any]] = []
    try:
        fc_draft = get_force_closed_cases(current_user, 'FORCE_CLOSED_DRAFT')
    except Exception:
        pass
    try:
        fc_complete = get_force_closed_cases(current_user, 'FORCE_CLOSED_COMPLETE')
    except Exception:
        pass

    generated_by = (
        current_user.display_name
        or current_user.username
        or 'النظام'
    )

    return generate_insight_word_report(
        grouped_inbox=grouped_inbox,
        fc_draft_cases=fc_draft,
        fc_complete_cases=fc_complete,
        generated_by=generated_by,
    )


def _apply_search_filter_to_inbox(
    grouped_inbox: List[Dict], search_term: str
) -> List[Dict]:
    """
    Filter grouped inbox subcases by a search term.
    Mirrors the filterBySearch function in InsightPage.jsx.
    """
    term = search_term.lower()
    result = []
    for unit in grouped_inbox:
        filtered = [
            s for s in unit.get('subcases', [])
            if (
                term in (s.get('incident_number') or '').lower()
                or term in str(s.get('incident_id') or '').lower()
                or term in (s.get('patient_name') or '').lower()
                or term in (s.get('case_description') or '').lower()
                or term in (s.get('category') or '').lower()
            )
        ]
        if filtered:
            result.append({**unit, 'subcases': filtered, 'pending_count': len(filtered)})
    return result


def _apply_scope_filter_to_subcases(subcases: List[Dict], current_user: CurrentUser) -> List[Dict]:
    """
    Filter subcases by user's allowed_unit_ids (Phase 2.5 scope engine).
    Same pattern as inbox_service._apply_scope_filter.
    
    Security-critical: Only returns subcases where target_org_unit_id is in allowed scope.
    
    Args:
        subcases: List of subcase dicts from DB layer
        current_user: User with allowed_unit_ids attribute
        
    Returns:
        Filtered list of subcases
    """
    if not subcases:
        return []
    
    # Get allowed unit IDs from Phase 2.5 Scope Engine
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # If no allowed_unit_ids, user has no scope - return empty
    if not allowed_unit_ids:
        return []
    
    # Filter subcases: keep only those where target_org_unit_id is in allowed scope
    filtered = []
    for subcase in subcases:
        target_org_unit_id = subcase.get('target_org_unit_id')
        status = subcase.get('status', '')
        
        # Skip force-closed cases (defensive filter)
        if status == 'FORCE_CLOSED':
            continue
        
        # Security check: only include if target is in allowed scope
        if target_org_unit_id in allowed_unit_ids:
            filtered.append(subcase)
    
    return filtered


def _group_subcases_by_org_unit(subcases: List[Dict]) -> Dict[int, List[Dict]]:
    """
    Group subcases by target_org_unit_id.
    
    Args:
        subcases: List of subcase dicts
        
    Returns:
        Dictionary: { org_unit_id: [subcase1, subcase2, ...] }
    """
    grouped = defaultdict(list)
    
    for subcase in subcases:
        org_unit_id = subcase.get('target_org_unit_id')
        if org_unit_id:
            grouped[org_unit_id].append(subcase)
    
    return dict(grouped)


def _build_section_group(org_unit_id: int, subcases: List[Dict]) -> Dict[str, Any]:
    """
    Build a section group object.
    
    Looks up supervisor name, counts subcases, sorts by waiting days.
    
    Args:
        org_unit_id: Organizational unit ID
        subcases: List of subcases for this org unit
        
    Returns:
        Group dict with metadata and sorted subcases
    """
    # Map numeric org_type codes to string enums
    # 323 = ADMINISTRATION, 324 = SECTION, 325 = DEPARTMENT
    ORG_TYPE_MAP = {
        323: 'ADMINISTRATION',
        324: 'SECTION',
        325: 'DEPARTMENT',
    }
    
    # Get org unit info from first subcase (all have same org unit)
    first_subcase = subcases[0] if subcases else {}
    org_unit_name = first_subcase.get('org_unit_name', f'Org Unit {org_unit_id}')
    raw_org_type = first_subcase.get('org_type')
    
    # Convert numeric org_type to string enum
    org_type = ORG_TYPE_MAP.get(raw_org_type, 'SECTION')  # Default to SECTION
    
    # Look up supervisor name
    supervisor_name = administrative_subcase_db.get_supervisor_name_for_org_unit(org_unit_id)
    if not supervisor_name:
        supervisor_name = "Unassigned"
    
    # Sort subcases by waiting_days DESC (longest waiting first)
    sorted_subcases = sorted(
        subcases,
        key=lambda s: s.get('waiting_days', 0),
        reverse=True
    )
    
    return {
        "section_id": org_unit_id,
        "section_name": org_unit_name,
        "org_type": org_type,
        "supervisor_name": supervisor_name,
        "pending_count": len(subcases),
        "subcases": sorted_subcases
    }
