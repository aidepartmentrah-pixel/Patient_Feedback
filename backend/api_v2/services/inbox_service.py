"""
Inbox Service — Phase 3 Workflow (API v2)

This service provides read-only inbox views for different administrative roles.
Each role sees subcases filtered by their organizational scope and current status.

ROLE MODEL:
- Section Administrator: Works on SUBMITTED_TO_SECTION
- Department Administrator: Works on SECTION_ACCEPTED_PENDING_DEPT
- Administration Administrator: Works on DEPT_ACCEPTED_PENDING_ADMIN

STATUS WORKFLOW:
SUBMITTED_TO_SECTION → SECTION_ACCEPTED_PENDING_DEPT → DEPT_ACCEPTED_PENDING_ADMIN

This service is READ-ONLY:
- No writes, no updates, no deletes
- No workflow transitions
- No side effects

Structure:
- Public functions: get_inbox, get_section_inbox, get_department_inbox, get_administration_inbox
- Internal helpers: _apply_scope_filter, _compute_allowed_actions
"""

from typing import List, Dict, Any, Optional
from api_v2.db_layer import administrative_subcase_db


# =============================================================================
# STATUS → ROLE RESPONSIBILITY MAP (MODEL A)
# =============================================================================

# Central authority for which statuses each role is responsible for.
# This map drives inbox routing and enforces strict responsibility model.
# Terminal statuses (ADMIN_APPROVED, SECTION_DENIED, FORCE_CLOSED, CLOSED)
# are EXCLUDED - they must not appear in any inbox.
STATUS_ROLE_MAP = {
    "SECTION_ADMIN": [
        "SUBMITTED_TO_SECTION",
        "RETURNED_TO_SECTION_FOR_REVISION"
    ],
    "DEPARTMENT_ADMIN": [
        "SECTION_ACCEPTED_PENDING_DEPT",
        "RETURNED_TO_DEPT_FOR_REVISION"
    ],
    "ADMINISTRATION_ADMIN": [
        "DEPT_ACCEPTED_PENDING_ADMIN"
    ],
    "COMPLAINT_SUPERVISOR": [
        "SECTION_DENIED"
    ]
}


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def get_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for current user (role-agnostic delegator).
    
    This is a thin routing function that delegates to the appropriate
    role-specific inbox function based on the user's role.
    
    This function exists solely for API v2 workflow router simplification.
    It contains NO business logic, NO filtering, NO scoping.
    
    MODEL A - STRICT RESPONSIBILITY ROUTING:
    Each role receives inbox items ONLY for statuses they are responsible for.
    
    Role Routing:
    - SECTION_ADMIN: Section-level inbox (SUBMITTED_TO_SECTION only)
    - DEPARTMENT_ADMIN: Department-level inbox (SECTION_ACCEPTED_PENDING_DEPT only)
    - ADMINISTRATION_ADMIN: Administration-level inbox (DEPT_ACCEPTED_PENDING_ADMIN only)
    - All other roles: Empty inbox (no workflow responsibility)
    
    Args:
        current_user: User object with role attribute
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        Empty list if user has no role or unsupported role
    """
    # Defensive check: ensure user has scopes
    if not current_user or not hasattr(current_user, 'scopes') or not current_user.scopes:
        # Return empty inbox instead of crashing
        return []
    
    # Get user's primary role from first scope
    primary_role = current_user.scopes[0].role_code
    
    # WORKER HANDLING - Workers see SECTION_DENIED subcases (same as COMPLAINT_SUPERVISOR)
    # They can reopen denied cases and resend to section
    if primary_role == 'WORKER':
        return get_worker_inbox(current_user)
    
    try:
        # STRICT RESPONSIBILITY MODEL - Only responsible roles get inbox items
        if primary_role == 'SECTION_ADMIN':
            return get_section_inbox(current_user)
        elif primary_role == 'DEPARTMENT_ADMIN':
            return get_department_inbox(current_user)
        elif primary_role == 'ADMINISTRATION_ADMIN':
            return get_administration_inbox(current_user)
        elif primary_role == 'COMPLAINT_SUPERVISOR':
            return get_complaint_supervisor_inbox(current_user)
        else:
            # All other roles (SOFTWARE_ADMIN, etc.)
            # return empty inbox - no workflow responsibility under Model A
            return []
    except Exception as e:
        # Log the error but return empty inbox to prevent frontend crash
        print(f"⚠️ Error in get_inbox for role {primary_role}: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty instead of raising to prevent network errors
        return []


def get_section_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Section Administrator role.
    
    Returns subcases with statuses defined in STATUS_ROLE_MAP for SECTION_ADMIN.
    Statuses are: SUBMITTED_TO_SECTION, RETURNED_TO_SECTION_FOR_REVISION
    
    Args:
        current_user: User object with role, section_id, department_id attributes
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user is not a Section Administrator
    """
    # Light role assertion using scopes
    if not current_user.scopes or current_user.scopes[0].role_code != 'SECTION_ADMIN':
        raise ValueError("User must be a Section Administrator to access section inbox")
    
    # Get subcases pending for section from DB layer
    # DB layer already queries for statuses in STATUS_ROLE_MAP["SECTION_ADMIN"]
    subcases = administrative_subcase_db.get_subcases_pending_for_section()
    
    # =========================================================================
    # SECURITY LOCK — Scope filtering MUST NOT be removed or bypassed
    # =========================================================================
    # Filters inbox items by allowed_unit_ids from Phase 2.5 Org Tree Scoping Engine.
    # Only subcases where TargetOrgUnitID is in current_user.allowed_unit_ids are returned.
    # This is the ONLY authority for data access - role does NOT grant data access.
    # Required for multi-tenant organizational security boundaries.
    # =========================================================================
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_department_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Department Administrator role.
    
    Returns subcases with statuses defined in STATUS_ROLE_MAP for DEPARTMENT_ADMIN.
    Statuses are: SECTION_ACCEPTED_PENDING_DEPT, RETURNED_TO_DEPT_FOR_REVISION
    
    Args:
        current_user: User object with role, section_id, department_id attributes
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user is not a Department Administrator
    """
    # Light role assertion using scopes
    if not current_user.scopes or current_user.scopes[0].role_code != 'DEPARTMENT_ADMIN':
        raise ValueError("User must be a Department Administrator to access department inbox")
    
    # Get subcases pending for department from DB layer
    # DB layer already queries for statuses in STATUS_ROLE_MAP["DEPARTMENT_ADMIN"]
    subcases = administrative_subcase_db.get_subcases_pending_for_department()
    
    # =========================================================================
    # SECURITY LOCK — Scope filtering MUST NOT be removed or bypassed
    # =========================================================================
    # Filters inbox items by allowed_unit_ids from Phase 2.5 Org Tree Scoping Engine.
    # Only subcases where TargetOrgUnitID is in current_user.allowed_unit_ids are returned.
    # This is the ONLY authority for data access - role does NOT grant data access.
    # Required for multi-tenant organizational security boundaries.
    # =========================================================================
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_administration_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Administration Administrator role.
    
    Returns subcases with statuses defined in STATUS_ROLE_MAP for ADMINISTRATION_ADMIN.
    Status is: DEPT_ACCEPTED_PENDING_ADMIN
    
    Args:
        current_user: User object with role, section_id, department_id attributes
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
        
    Raises:
        ValueError: If user is not an Administration Administrator
    """
    # Light role assertion using scopes
    if not current_user.scopes or current_user.scopes[0].role_code != 'ADMINISTRATION_ADMIN':
        raise ValueError("User must be an Administration Administrator to access administration inbox")
    
    # Get subcases pending for administration from DB layer
    # DB layer already queries for statuses in STATUS_ROLE_MAP["ADMINISTRATION_ADMIN"]
    subcases = administrative_subcase_db.get_subcases_pending_for_administration()
    
    # =========================================================================
    # SECURITY LOCK — Scope filtering MUST NOT be removed or bypassed
    # =========================================================================
    # Filters inbox items by allowed_unit_ids from Phase 2.5 Org Tree Scoping Engine.
    # Only subcases where TargetOrgUnitID is in current_user.allowed_unit_ids are returned.
    # This is the ONLY authority for data access - role does NOT grant data access.
    # Required for multi-tenant organizational security boundaries.
    # =========================================================================
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_complaint_supervisor_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Complaint Supervisor role.
    
    Returns subcases that the section has DENIED (rejected responsibility for).
    The COMPLAINT_SUPERVISOR can review these, optionally edit the incident
    severity, and reopen the case by returning it to the section for revision.
    
    Status: SECTION_DENIED
    
    Args:
        current_user: User object with COMPLAINT_SUPERVISOR role
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
    """
    # Get subcases denied by section
    subcases = administrative_subcase_db.get_subcases_denied_by_section()
    
    # =========================================================================
    # SECURITY LOCK — Scope filtering MUST NOT be removed or bypassed
    # =========================================================================
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_worker_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get inbox for Worker role.
    
    Returns subcases that the section has DENIED (rejected responsibility for).
    Workers can review and reopen denied cases (resend to section).
    
    Status: SECTION_DENIED
    
    Args:
        current_user: User object with WORKER role
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
    """
    # Get subcases denied by section
    subcases = administrative_subcase_db.get_subcases_denied_by_section()
    
    # SECURITY LOCK — Scope filtering
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


def get_unified_inbox(current_user) -> List[Dict[str, Any]]:
    """
    Get unified inbox for full-access roles (SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR, WORKER).
    
    Returns ALL pending subcases across all workflow levels:
    - SUBMITTED_TO_SECTION (section level)
    - SECTION_ACCEPTED_PENDING_DEPT (department level)
    - DEPT_ACCEPTED_PENDING_ADMIN (administration level)
    
    These roles have supervisory/administrative access and can view the entire
    workflow pipeline. Scope filtering is still applied via Phase 2.5 engine.
    
    Args:
        current_user: User object with full-access role
        
    Returns:
        List of inbox items (dictionaries) with allowed actions
    """
    # Get subcases from all workflow levels
    section_subcases = administrative_subcase_db.get_subcases_pending_for_section()
    dept_subcases = administrative_subcase_db.get_subcases_pending_for_department()
    admin_subcases = administrative_subcase_db.get_subcases_pending_for_administration()
    
    # Combine all subcases
    all_subcases = section_subcases + dept_subcases + admin_subcases
    
    # Apply scope filter (Phase 2.5 engine will filter based on allowed_unit_ids)
    filtered_subcases = _apply_scope_filter(all_subcases, current_user)
    
    # Build inbox items with allowed actions
    inbox_items = []
    for subcase in filtered_subcases:
        item = _build_inbox_item(subcase, current_user)
        inbox_items.append(item)
    
    return inbox_items


# =============================================================================
# ARCHIVE PUBLIC FUNCTIONS
# =============================================================================
# Archive shows subcases that have moved PAST a user's workflow stage.
# These are read-only (view action only) - no workflow actions allowed.
# =============================================================================

def get_archive(current_user) -> List[Dict[str, Any]]:
    """
    Get archive for current user (role-agnostic delegator).
    
    Archive contains subcases that the user previously processed,
    which have now moved past their workflow stage.
    
    Unlike inbox (active items requiring action), archive shows
    completed/processed items for reference only.
    
    Role Routing:
    - SECTION_ADMIN: Cases that moved past section stage
    - DEPARTMENT_ADMIN: Cases that moved past department stage
    - ADMINISTRATION_ADMIN: Cases that reached terminal status
    - COMPLAINT_SUPERVISOR: Cases that were reopened
    - WORKER: Cases that were reopened
    - All other roles: Empty archive
    
    Args:
        current_user: User object with role attribute
        
    Returns:
        List of archive items (view-only) or empty list
    """
    # Defensive check: ensure user has scopes
    if not current_user or not hasattr(current_user, 'scopes') or not current_user.scopes:
        return []
    
    # Get user's primary role from first scope
    primary_role = current_user.scopes[0].role_code
    
    try:
        if primary_role == 'SECTION_ADMIN':
            return get_section_archive(current_user)
        elif primary_role == 'DEPARTMENT_ADMIN':
            return get_department_archive(current_user)
        elif primary_role == 'ADMINISTRATION_ADMIN':
            return get_administration_archive(current_user)
        elif primary_role == 'COMPLAINT_SUPERVISOR':
            return get_complaint_supervisor_archive(current_user)
        elif primary_role == 'WORKER':
            return get_worker_archive(current_user)
        else:
            # All other roles have no archive
            return []
    except Exception as e:
        print(f"⚠️ Error in get_archive for role {primary_role}: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def get_section_archive(current_user) -> List[Dict[str, Any]]:
    """
    Get archive for Section Administrator.
    
    Returns subcases that the section admin has processed and that have
    moved past the section stage. These are view-only.
    
    Args:
        current_user: User object with SECTION_ADMIN role
        
    Returns:
        List of archive items (view-only)
    """
    # Get subcases that have moved past section stage
    subcases = administrative_subcase_db.get_subcases_archived_for_section()
    
    # SECURITY LOCK — Scope filtering still applies
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build archive items (view-only)
    archive_items = []
    for subcase in filtered_subcases:
        item = _build_archive_item(subcase)
        archive_items.append(item)
    
    return archive_items


def get_department_archive(current_user) -> List[Dict[str, Any]]:
    """
    Get archive for Department Administrator.
    
    Returns subcases that have moved past the department stage.
    
    Args:
        current_user: User object with DEPARTMENT_ADMIN role
        
    Returns:
        List of archive items (view-only)
    """
    # Get subcases that have moved past department stage
    subcases = administrative_subcase_db.get_subcases_archived_for_department()
    
    # SECURITY LOCK — Scope filtering
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build archive items
    archive_items = []
    for subcase in filtered_subcases:
        item = _build_archive_item(subcase)
        archive_items.append(item)
    
    return archive_items


def get_administration_archive(current_user) -> List[Dict[str, Any]]:
    """
    Get archive for Administration Administrator.
    
    Returns subcases that have reached terminal status.
    
    Args:
        current_user: User object with ADMINISTRATION_ADMIN role
        
    Returns:
        List of archive items (view-only)
    """
    # Get subcases that have moved past administration stage
    subcases = administrative_subcase_db.get_subcases_archived_for_administration()
    
    # SECURITY LOCK — Scope filtering
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build archive items
    archive_items = []
    for subcase in filtered_subcases:
        item = _build_archive_item(subcase)
        archive_items.append(item)
    
    return archive_items


def get_complaint_supervisor_archive(current_user) -> List[Dict[str, Any]]:
    """
    Get archive for Complaint Supervisor.
    
    Returns subcases that were reopened after section denial.
    
    Args:
        current_user: User object with COMPLAINT_SUPERVISOR role
        
    Returns:
        List of archive items (view-only)
    """
    # Get subcases that were reopened
    subcases = administrative_subcase_db.get_subcases_archived_for_complaint_supervisor()
    
    # SECURITY LOCK — Scope filtering
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build archive items
    archive_items = []
    for subcase in filtered_subcases:
        item = _build_archive_item(subcase)
        archive_items.append(item)
    
    return archive_items


def get_worker_archive(current_user) -> List[Dict[str, Any]]:
    """
    Get archive for Worker.
    
    Returns subcases that were reopened after section denial.
    Same as COMPLAINT_SUPERVISOR archive.
    
    Args:
        current_user: User object with WORKER role
        
    Returns:
        List of archive items (view-only)
    """
    # Get subcases that were reopened
    subcases = administrative_subcase_db.get_subcases_archived_for_complaint_supervisor()
    
    # SECURITY LOCK — Scope filtering
    filtered_subcases = _apply_scope_filter(subcases, current_user)
    
    # Build archive items
    archive_items = []
    for subcase in filtered_subcases:
        item = _build_archive_item(subcase)
        archive_items.append(item)
    
    return archive_items


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _apply_scope_filter(subcases: List[Any], current_user) -> List[Any]:
    """
    Filter subcases based on user's organizational scope from Phase 2.5 Scope Engine.
    
    SECURITY-CRITICAL: This function enforces the central scope boundary.
    Only subcases where TargetOrgUnitID is in current_user.allowed_unit_ids are returned.
    
    This is the ONLY authority for data access:
    - Role does NOT grant data access
    - Legacy local org IDs are IGNORED
    - current_user.allowed_unit_ids is the single source of truth
    
    Even if role logic, router, or frontend is compromised, this function ensures
    no out-of-scope data is ever returned.
    
    Args:
        subcases: List of subcase dicts from DB layer
        current_user: User object with allowed_unit_ids attribute (set[int])
        
    Returns:
        Filtered list of subcases where target_org_unit_id is in allowed_unit_ids
    """
    if not subcases:
        return []
    
    # Get allowed unit IDs from Phase 2.5 Scope Engine
    # This is the ONLY authority for data access
    allowed_unit_ids = getattr(current_user, 'allowed_unit_ids', None)
    
    # If no allowed_unit_ids, user has no scope - return empty
    if not allowed_unit_ids:
        return []
    
    # Filter subcases: keep only those where target_org_unit_id is in allowed scope
    # ALSO exclude FORCE_CLOSED cases (defensive filter)
    filtered = []
    for subcase in subcases:
        # Subcases from DB layer are dicts, not objects
        target_org_unit_id = subcase.get('target_org_unit_id')
        status = subcase.get('status', '')
        
        # Skip force-closed cases (shouldn't appear in inbox queries, but defensive)
        if status == 'FORCE_CLOSED':
            continue
        
        # Security check: only include if target is in allowed scope
        if target_org_unit_id in allowed_unit_ids:
            filtered.append(subcase)
    
    return filtered


def _compute_allowed_actions(subcase: Dict[str, Any], current_user) -> List[str]:
    """
    Compute allowed actions for a subcase based on its status and user's role.
    
    MODEL A - STRICT RESPONSIBILITY MATRIX:
    Actions are determined ONLY by exact (role, status) combination.
    No supervisory override. No cross-stage actions.
    
    Returns lowercase action codes that frontend uses to render buttons:
    - "view": Navigate to case detail page
    - "submit_response": Submit explanation + action items (Section Admin only)
    - "accept": Approve and advance workflow to next level
    - "reject": Reject case with reason
    
    STRICT ACTION MATRIX:
    
    SECTION_ADMIN:
      - SUBMITTED_TO_SECTION: ["view", "submit_response", "reject"]
      - RETURNED_TO_SECTION_FOR_REVISION: ["view", "submit_response", "reject"]
      - All other statuses: ["view"] only
    
    DEPARTMENT_ADMIN:
      - SECTION_ACCEPTED_PENDING_DEPT: ["view", "accept", "reject"]
      - RETURNED_TO_DEPT_FOR_REVISION: ["view", "accept", "reject"]
      - All other statuses: ["view"] only
    
    ADMINISTRATION_ADMIN:
      - DEPT_ACCEPTED_PENDING_ADMIN: ["view", "accept", "reject"]
      - All other statuses: ["view"] only
    
    SOFTWARE_ADMIN:
      - DEPT_ACCEPTED_PENDING_ADMIN: ["view", "accept", "reject"] (final-stage authority only)
      - All other statuses: ["view"] only (no section/dept override)
    
    ALL OTHER ROLES (WORKER, COMPLAINT_SUPERVISOR, etc.):
      - All statuses: ["view"] only
    
    Args:
        subcase: Subcase dict from DB layer
        current_user: CurrentUser object with .scopes[] array
        
    Returns:
        List of lowercase action strings (e.g., ["view", "submit_response", "reject"])
    """
    # DB layer returns dict with lowercase snake_case keys
    status = subcase.get('status', None)
    
    # CurrentUser has .scopes[] array with role_code
    if not current_user or not hasattr(current_user, 'scopes') or not current_user.scopes:
        # Defensive: if no role, only allow view
        return ["view"]
    
    role_code = current_user.scopes[0].role_code
    
    # Helper: a response has been submitted if any explanation text is present
    _has_response = bool(
        subcase.get('section_explanation_text')
        or subcase.get('department_explanation_text')
        or subcase.get('administration_explanation_text')
    )

    # STRICT MATRIX - Actions based on exact (role, status) match
    
    # SECTION_ADMIN actions
    if role_code == 'SECTION_ADMIN':
        if status in ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION']:
            actions = ["view", "submit_response", "reject"]
            # Section admin can re-view their previous response when
            # the case has been returned for revision.
            if status == 'RETURNED_TO_SECTION_FOR_REVISION' and _has_response:
                actions.append("view_response")
            return actions
        else:
            return ["view"]
    
    # DEPARTMENT_ADMIN actions
    elif role_code == 'DEPARTMENT_ADMIN':
        if status in ['SECTION_ACCEPTED_PENDING_DEPT', 'RETURNED_TO_DEPT_FOR_REVISION']:
            actions = ["view", "accept", "reject"]
            if _has_response:
                actions.append("view_response")
            return actions
        else:
            return ["view"]
    
    # ADMINISTRATION_ADMIN actions
    elif role_code == 'ADMINISTRATION_ADMIN':
        if status == 'DEPT_ACCEPTED_PENDING_ADMIN':
            actions = ["view", "accept", "reject"]
            if _has_response:
                actions.append("view_response")
            return actions
        else:
            return ["view"]
    
    # SOFTWARE_ADMIN actions - Final-stage authority only, no section/dept override
    elif role_code == 'SOFTWARE_ADMIN':
        if status == 'DEPT_ACCEPTED_PENDING_ADMIN':
            actions = ["view", "accept", "reject"]
            if _has_response:
                actions.append("view_response")
            return actions
        else:
            return ["view"]
    
    # COMPLAINT_SUPERVISOR actions
    elif role_code == 'COMPLAINT_SUPERVISOR':
        if status == 'SECTION_DENIED':
            actions = ["view", "reopen"]
            if bool(
                subcase.get('section_explanation_text')
                or subcase.get('section_rejection_text')
            ):
                actions.append("view_response")
            return actions
        else:
            return ["view"]
    
    # WORKER actions - same as COMPLAINT_SUPERVISOR for SECTION_DENIED
    elif role_code == 'WORKER':
        if status == 'SECTION_DENIED':
            actions = ["view", "reopen"]
            if bool(
                subcase.get('section_explanation_text')
                or subcase.get('section_rejection_text')
            ):
                actions.append("view_response")
            return actions
        else:
            return ["view"]
    
    # All other roles
    # have view-only access - no workflow actions
    else:
        return ["view"]


def _build_inbox_item(subcase: Dict[str, Any], current_user) -> Dict[str, Any]:
    """
    Build an inbox item dictionary from a subcase dict.
    
    Converts DB layer dict to API-ready dictionary format.
    
    Args:
        subcase: Subcase dict from DB layer
        current_user: User object for computing allowed actions
        
    Returns:
        Dictionary with inbox item structure
    """
    # Compute allowed actions for this subcase
    allowed_actions = _compute_allowed_actions(subcase, current_user)
    
    # Build the inbox item (subcase is already a dict from DB layer)
    inbox_item = {
        "subcase_id": subcase.get('subcase_id'),
        "case_type": subcase.get('case_type'),
        "incident_id": subcase.get('incident_request_case_id'),
        "seasonal_report_id": subcase.get('seasonal_report_id'),
        "target_org_unit_id": subcase.get('target_org_unit_id'),
        "target_org_unit_name": subcase.get('org_unit_name'),
        "status": subcase.get('status'),
        "created_at": subcase.get('created_at'),
        "allowed_actions": allowed_actions
    }
    
    return inbox_item


def _build_archive_item(subcase: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build an archive item dictionary from a subcase dict.
    
    Archive items are READ-ONLY - they only have 'view' action.
    No workflow actions are allowed on archived items.
    
    Args:
        subcase: Subcase dict from DB layer
        
    Returns:
        Dictionary with archive item structure (view-only)
    """
    # Archive items are always view-only
    archive_item = {
        "subcase_id": subcase.get('subcase_id'),
        "case_type": subcase.get('case_type'),
        "incident_id": subcase.get('incident_request_case_id'),
        "seasonal_report_id": subcase.get('seasonal_report_id'),
        "target_org_unit_id": subcase.get('target_org_unit_id'),
        "target_org_unit_name": subcase.get('org_unit_name'),
        "status": subcase.get('status'),
        "created_at": subcase.get('created_at'),
        "updated_at": subcase.get('updated_at'),  # Include when it was processed
        "allowed_actions": ["view"]  # Archive is read-only
    }
    
    return archive_item
