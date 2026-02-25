"""
Phase 2.5.4 - Scope Guards Usage Examples
Shows how the scope guards will be used in endpoints
"""

# =====================================================
# Example 1: Single Unit Access
# =====================================================
"""
from fastapi import APIRouter, Depends
from api.dependencies.user_context import get_current_user
from api.schemas.auth_models import CurrentUser
from api.utils.guards import require_unit_in_scope

router = APIRouter()

@router.get("/department/{dept_id}/stats")
def get_department_stats(
    dept_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    '''Get statistics for a specific department'''
    
    # Enforce organizational scope
    require_unit_in_scope(current_user, dept_id)
    
    # If we reach here, user has access to this department
    stats = compute_department_stats(dept_id)
    return stats
"""

# =====================================================
# Example 2: Multi-Unit Query
# =====================================================
"""
@router.post("/reports/multi-unit")
def generate_multi_unit_report(
    requested_units: list[int],
    current_user: CurrentUser = Depends(get_current_user)
):
    '''Generate a report spanning multiple org units'''
    
    # Ensure user has access to at least one requested unit
    require_any_unit_in_scope(current_user, requested_units)
    
    # Filter to only units the user can actually access
    accessible_units = [
        unit_id for unit_id in requested_units
        if unit_id in current_user.allowed_unit_ids
    ]
    
    # Generate report for accessible units only
    report = generate_report(accessible_units)
    return report
"""

# =====================================================
# Example 3: Dashboard with Dynamic Scope
# =====================================================
"""
@router.get("/dashboard")
def get_dashboard(
    org_unit_id: int | None = None,
    current_user: CurrentUser = Depends(get_current_user)
):
    '''Get dashboard data for an org unit'''
    
    # If no unit specified, use user's full allowed scope
    if org_unit_id is None:
        allowed_units = current_user.allowed_unit_ids
    else:
        # If unit specified, verify user has access
        require_unit_in_scope(current_user, org_unit_id)
        allowed_units = {org_unit_id}
    
    # Fetch data scoped to allowed units
    dashboard_data = fetch_dashboard_data(allowed_units)
    return dashboard_data
"""

# =====================================================
# Example 4: Incident Access
# =====================================================
"""
@router.get("/incidents/{incident_id}")
def get_incident(
    incident_id: int,
    current_user: CurrentUser = Depends(get_current_user)
):
    '''Get a specific incident'''
    
    # Fetch incident
    incident = fetch_incident(incident_id)
    
    # Verify user has access to the incident's org unit
    require_unit_in_scope(current_user, incident.org_unit_id)
    
    # Return incident data
    return incident
"""

# =====================================================
# Example 5: Batch Operations with Scope Filtering
# =====================================================
"""
@router.post("/incidents/batch-update")
def batch_update_incidents(
    incident_ids: list[int],
    updates: dict,
    current_user: CurrentUser = Depends(get_current_user)
):
    '''Update multiple incidents'''
    
    # Fetch all incidents
    incidents = fetch_incidents(incident_ids)
    
    # Get org units involved
    org_units = {inc.org_unit_id for inc in incidents}
    
    # Verify user has access to at least one
    require_any_unit_in_scope(current_user, org_units)
    
    # Filter to only incidents user can access
    accessible_incidents = [
        inc for inc in incidents
        if inc.org_unit_id in current_user.allowed_unit_ids
    ]
    
    # Perform updates only on accessible incidents
    result = update_incidents(accessible_incidents, updates)
    return result
"""

print("=" * 60)
print("SCOPE GUARDS - USAGE PATTERNS")
print("=" * 60)

print("\n✅ Pattern 1: Single Unit Enforcement")
print("   require_unit_in_scope(current_user, unit_id)")
print("   → Use when endpoint accesses ONE specific org unit")
print("   → Raises 403 if user doesn't have access")

print("\n✅ Pattern 2: Multi-Unit Enforcement")
print("   require_any_unit_in_scope(current_user, unit_ids)")
print("   → Use when endpoint needs access to AT LEAST ONE unit")
print("   → Raises 403 if user has no access to any unit")
print("   → Then filter to accessible units only")

print("\n✅ Pattern 3: Dynamic Scope")
print("   Use current_user.allowed_unit_ids directly")
print("   → When fetching all data in user's scope")
print("   → No guard needed, just use the allowed set")

print("\n✅ Pattern 4: Verify After Fetch")
print("   Fetch resource, then check its org_unit_id")
print("   → Useful when org_unit_id is in the resource")
print("   → Prevents unauthorized data leakage")

print("\n✅ Pattern 5: Filter Collections")
print("   Check access, then filter to accessible items")
print("   → For batch operations")
print("   → Ensures only authorized items are modified")

print("\n" + "=" * 60)
print("GUARD CHARACTERISTICS")
print("=" * 60)

print("\n✓ Simple functions (not decorators)")
print("✓ Explicit enforcement (call where needed)")
print("✓ Fail-fast (raises HTTPException)")
print("✓ No side effects")
print("✓ No DB access")
print("✓ No scope computation")
print("✓ Only checks current_user.allowed_unit_ids")

print("\n" + "=" * 60)
print("✅ Scope guards ready for use in endpoints")
print("=" * 60)
