"""
Scope Resolver

Computes the effective org-unit scope for the current user.
This module transforms the user's role-scope configuration into a concrete set of 
allowed org unit IDs based on the organizational hierarchy.

Design:
- Takes CurrentUser as input (no additional DB queries)
- Returns set[int] of allowed org unit IDs
- SOFTWARE_ADMIN gets all org units
- Normal users must have exactly ONE scope
- Scope expansion is based on org_unit_type (SECTION/DEPARTMENT/ADMINISTRATION)
- Pure, deterministic logic with no side effects
"""

from typing import Set
from ..schemas.auth_models import CurrentUser
from . import org_tree_service


# =========================================================
# CONSTANTS
# =========================================================

SOFTWARE_ADMIN = "SOFTWARE_ADMIN"

# Org unit types
ORG_TYPE_SECTION = "SECTION"
ORG_TYPE_DEPARTMENT = "DEPARTMENT"
ORG_TYPE_ADMINISTRATION = "ADMINISTRATION"


# =========================================================
# SCOPE RESOLUTION
# =========================================================

def resolve_user_scope(current_user: CurrentUser) -> Set[int]:
    """
    Return the set of org unit IDs this user is allowed to access.
    
    Algorithm:
    1. If user has SOFTWARE_ADMIN role in any scope → return ALL org units
    2. Otherwise, user must have exactly ONE scope (else raise exception)
    3. Expand based on org_unit_type:
       - SECTION → only that section ID
       - DEPARTMENT → department + all descendants
       - ADMINISTRATION → administration + all descendants
    
    Args:
        current_user: The authenticated user with role-scope assignments
        
    Returns:
        Set[int]: Set of org unit IDs the user is allowed to access
        
    Raises:
        ValueError: If user has invalid scope configuration (not exactly 1 scope)
    """
    
    # =====================================================
    # Step 1: Check for SOFTWARE_ADMIN
    # =====================================================
    for scope in current_user.scopes:
        if scope.role_code == SOFTWARE_ADMIN:
            # SOFTWARE_ADMIN gets access to ALL org units
            full_tree = org_tree_service.get_full_tree()
            all_unit_ids = {unit["UniqueID"] for unit in full_tree}
            return all_unit_ids
    
    # =====================================================
    # Step 2: Validate normal user has exactly ONE scope
    # =====================================================
    if len(current_user.scopes) != 1:
        raise ValueError(
            f"User has invalid scope configuration: exactly one scope is required. "
            f"User '{current_user.username}' has {len(current_user.scopes)} scopes."
        )
    
    # =====================================================
    # Step 3: Get the single scope
    # =====================================================
    user_scope = current_user.scopes[0]
    org_unit_id = user_scope.org_unit_id
    org_unit_type = user_scope.org_unit_type.upper()
    
    # =====================================================
    # Step 4: Expand based on org_unit_type
    # =====================================================
    
    if org_unit_type == ORG_TYPE_SECTION:
        # Section users can only access their own section
        return {org_unit_id}
    
    elif org_unit_type == ORG_TYPE_DEPARTMENT:
        # Department users can access the department and all its sections
        return org_tree_service.get_descendants(org_unit_id)
    
    elif org_unit_type == ORG_TYPE_ADMINISTRATION:
        # Administration users can access the administration and all its children
        return org_tree_service.get_descendants(org_unit_id)
    
    else:
        # Unknown org unit type - treat as single unit for safety
        # This should not happen in production but provides a safe fallback
        return {org_unit_id}


def is_allowed_unit(current_user: CurrentUser, org_unit_id: int) -> bool:
    """
    Check if the current user is allowed to access a specific org unit.
    
    Args:
        current_user: The authenticated user
        org_unit_id: The org unit ID to check
        
    Returns:
        bool: True if the user is allowed to access this org unit
    """
    allowed_units = resolve_user_scope(current_user)
    return org_unit_id in allowed_units


# =========================================================
# TESTING / DEBUG
# =========================================================

if __name__ == "__main__":
    """
    Manual test block to verify scope resolution logic.
    """
    from ..schemas.auth_models import UserScope
    
    print("=" * 60)
    print("SCOPE RESOLVER - TEST")
    print("=" * 60)
    
    # Load tree to find test org units
    full_tree = org_tree_service.get_full_tree()
    
    # Find test units
    admin_nodes = [n for n in full_tree if n["ParentID"] == n["UniqueID"]]
    if not admin_nodes:
        print("❌ No administration nodes found in tree")
        exit(1)
    
    admin_node = admin_nodes[0]
    admin_id = admin_node["UniqueID"]
    admin_name = admin_node["Name"]
    
    dept_nodes = [n for n in full_tree 
                  if n["ParentID"] == admin_id and n["UniqueID"] != admin_id]
    if not dept_nodes:
        print("❌ No department nodes found")
        exit(1)
    
    dept_node = dept_nodes[0]
    dept_id = dept_node["UniqueID"]
    dept_name = dept_node["Name"]
    
    section_nodes = [n for n in full_tree 
                     if n["ParentID"] == dept_id and n["UniqueID"] != dept_id]
    if not section_nodes:
        print("❌ No section nodes found")
        exit(1)
    
    section_node = section_nodes[0]
    section_id = section_node["UniqueID"]
    section_name = section_node["Name"]
    
    print(f"\n📊 Test Org Units:")
    print(f"   Administration: {admin_name} (ID: {admin_id})")
    print(f"   Department: {dept_name} (ID: {dept_id})")
    print(f"   Section: {section_name} (ID: {section_id})")
    
    # =====================================================
    # Test 1: SOFTWARE_ADMIN gets all units
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 1: SOFTWARE_ADMIN Scope")
    print("=" * 60)
    
    software_admin_user = CurrentUser(
        user_id=1,
        username="software_admin",
        is_active=True,
        scopes=[UserScope(
            role_code="SOFTWARE_ADMIN",
            org_unit_id=0,
            org_unit_type="ADMINISTRATION"
        )]
    )
    
    admin_scope = resolve_user_scope(software_admin_user)
    print(f"✓ SOFTWARE_ADMIN scope: {len(admin_scope)} org units")
    print(f"  Expected: All {len(full_tree)} units")
    print(f"  Result: {'✅ PASS' if len(admin_scope) == len(full_tree) else '❌ FAIL'}")
    
    # =====================================================
    # Test 2: Section user gets only their section
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 2: SECTION Scope")
    print("=" * 60)
    
    section_user = CurrentUser(
        user_id=2,
        username="section_admin",
        is_active=True,
        scopes=[UserScope(
            role_code="SECTION_ADMIN",
            org_unit_id=section_id,
            org_unit_type="SECTION"
        )]
    )
    
    section_scope = resolve_user_scope(section_user)
    print(f"✓ SECTION scope: {len(section_scope)} org units")
    print(f"  Expected: 1 unit (only the section)")
    print(f"  Result: {'✅ PASS' if section_scope == {section_id} else '❌ FAIL'}")
    
    # =====================================================
    # Test 3: Department user gets department + sections
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 3: DEPARTMENT Scope")
    print("=" * 60)
    
    dept_user = CurrentUser(
        user_id=3,
        username="dept_admin",
        is_active=True,
        scopes=[UserScope(
            role_code="DEPARTMENT_ADMIN",
            org_unit_id=dept_id,
            org_unit_type="DEPARTMENT"
        )]
    )
    
    dept_scope = resolve_user_scope(dept_user)
    expected_dept_descendants = org_tree_service.get_descendants(dept_id)
    print(f"✓ DEPARTMENT scope: {len(dept_scope)} org units")
    print(f"  Expected: {len(expected_dept_descendants)} units (department + sections)")
    print(f"  Result: {'✅ PASS' if dept_scope == expected_dept_descendants else '❌ FAIL'}")
    
    # =====================================================
    # Test 4: Administration user gets administration + all children
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 4: ADMINISTRATION Scope")
    print("=" * 60)
    
    admin_user = CurrentUser(
        user_id=4,
        username="admin_admin",
        is_active=True,
        scopes=[UserScope(
            role_code="ADMINISTRATION_ADMIN",
            org_unit_id=admin_id,
            org_unit_type="ADMINISTRATION"
        )]
    )
    
    admin_user_scope = resolve_user_scope(admin_user)
    expected_admin_descendants = org_tree_service.get_descendants(admin_id)
    print(f"✓ ADMINISTRATION scope: {len(admin_user_scope)} org units")
    print(f"  Expected: {len(expected_admin_descendants)} units (administration + all children)")
    print(f"  Result: {'✅ PASS' if admin_user_scope == expected_admin_descendants else '❌ FAIL'}")
    
    # =====================================================
    # Test 5: User with 0 scopes should raise error
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 5: Invalid Scope Configuration (0 scopes)")
    print("=" * 60)
    
    no_scope_user = CurrentUser(
        user_id=5,
        username="no_scope_user",
        is_active=True,
        scopes=[]
    )
    
    try:
        resolve_user_scope(no_scope_user)
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✅ PASS: Correctly raised error: {e}")
    
    # =====================================================
    # Test 6: User with 2 scopes should raise error
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 6: Invalid Scope Configuration (2 scopes)")
    print("=" * 60)
    
    multi_scope_user = CurrentUser(
        user_id=6,
        username="multi_scope_user",
        is_active=True,
        scopes=[
            UserScope(
                role_code="SECTION_ADMIN",
                org_unit_id=section_id,
                org_unit_type="SECTION"
            ),
            UserScope(
                role_code="DEPARTMENT_ADMIN",
                org_unit_id=dept_id,
                org_unit_type="DEPARTMENT"
            )
        ]
    )
    
    try:
        resolve_user_scope(multi_scope_user)
        print("❌ FAIL: Should have raised ValueError")
    except ValueError as e:
        print(f"✅ PASS: Correctly raised error: {e}")
    
    # =====================================================
    # Test 7: is_allowed_unit helper function
    # =====================================================
    print("\n" + "=" * 60)
    print("Test 7: is_allowed_unit Helper Function")
    print("=" * 60)
    
    # Section user should only access their section
    section_can_access_section = is_allowed_unit(section_user, section_id)
    section_can_access_dept = is_allowed_unit(section_user, dept_id)
    
    print(f"✓ Section user can access own section: {section_can_access_section}")
    print(f"✓ Section user can access parent dept: {section_can_access_dept}")
    print(f"  Result: {'✅ PASS' if section_can_access_section and not section_can_access_dept else '❌ FAIL'}")
    
    # Department user should access department and sections
    dept_can_access_dept = is_allowed_unit(dept_user, dept_id)
    dept_can_access_section = is_allowed_unit(dept_user, section_id)
    
    print(f"✓ Dept user can access own dept: {dept_can_access_dept}")
    print(f"✓ Dept user can access child section: {dept_can_access_section}")
    print(f"  Result: {'✅ PASS' if dept_can_access_dept and dept_can_access_section else '❌ FAIL'}")
    
    print("\n" + "=" * 60)
    print("✓ All tests completed")
    print("=" * 60)
