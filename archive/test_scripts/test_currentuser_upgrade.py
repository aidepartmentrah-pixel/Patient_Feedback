"""
Test script for Phase 2.5.3 - CurrentUser Upgrade
Verifies that allowed_unit_ids is computed and attached to CurrentUser
"""

import sys
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.schemas.auth_models import CurrentUser, UserScope
from api.services.scope_resolver import resolve_user_scope
from api.services.org_tree_service import get_full_tree

print("=" * 60)
print("PHASE 2.5.3 - CURRENTUSER UPGRADE TEST")
print("=" * 60)

# Load tree to find test org units
full_tree = get_full_tree()
print(f"\n✓ Loaded org tree: {len(full_tree)} units")

# Find test units
admin_nodes = [n for n in full_tree if n["ParentID"] == n["UniqueID"]]
admin_node = admin_nodes[0]
admin_id = admin_node["UniqueID"]
admin_name = admin_node["Name"]

dept_nodes = [n for n in full_tree 
              if n["ParentID"] == admin_id and n["UniqueID"] != admin_id]
dept_node = dept_nodes[0]
dept_id = dept_node["UniqueID"]
dept_name = dept_node["Name"]

section_nodes = [n for n in full_tree 
                 if n["ParentID"] == dept_id and n["UniqueID"] != dept_id]
section_node = section_nodes[0]
section_id = section_node["UniqueID"]
section_name = section_node["Name"]

print(f"\n📊 Test Org Units:")
print(f"   Admin: {admin_name} (ID: {admin_id})")
print(f"   Dept:  {dept_name} (ID: {dept_id})")
print(f"   Sect:  {section_name} (ID: {section_id})")

# =====================================================
# Test 1: Section User - CurrentUser with allowed_unit_ids
# =====================================================
print("\n" + "=" * 60)
print("Test 1: Section User CurrentUser")
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

# Simulate what get_current_user_from_session does
section_user.allowed_unit_ids = resolve_user_scope(section_user)

print(f"✓ Username: {section_user.username}")
print(f"✓ User ID: {section_user.user_id}")
print(f"✓ Scopes: {len(section_user.scopes)} scope(s)")
print(f"✓ Allowed Unit IDs: {section_user.allowed_unit_ids}")
print(f"  Expected: {{{section_id}}}")
print(f"  Result: {'✅ PASS' if section_user.allowed_unit_ids == {section_id} else '❌ FAIL'}")

# =====================================================
# Test 2: Department User
# =====================================================
print("\n" + "=" * 60)
print("Test 2: Department User CurrentUser")
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

dept_user.allowed_unit_ids = resolve_user_scope(dept_user)

print(f"✓ Username: {dept_user.username}")
print(f"✓ User ID: {dept_user.user_id}")
print(f"✓ Scopes: {len(dept_user.scopes)} scope(s)")
print(f"✓ Allowed Unit IDs: {len(dept_user.allowed_unit_ids)} units")
print(f"  Should include: dept ({dept_id}) + sections")
print(f"  Result: {'✅ PASS' if dept_id in dept_user.allowed_unit_ids and section_id in dept_user.allowed_unit_ids else '❌ FAIL'}")

# =====================================================
# Test 3: Administration User
# =====================================================
print("\n" + "=" * 60)
print("Test 3: Administration User CurrentUser")
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

admin_user.allowed_unit_ids = resolve_user_scope(admin_user)

print(f"✓ Username: {admin_user.username}")
print(f"✓ User ID: {admin_user.user_id}")
print(f"✓ Scopes: {len(admin_user.scopes)} scope(s)")
print(f"✓ Allowed Unit IDs: {len(admin_user.allowed_unit_ids)} units")
print(f"  Should include: admin ({admin_id}) + all children")
print(f"  Result: {'✅ PASS' if admin_id in admin_user.allowed_unit_ids and dept_id in admin_user.allowed_unit_ids and section_id in admin_user.allowed_unit_ids else '❌ FAIL'}")

# =====================================================
# Test 4: SOFTWARE_ADMIN - All Units
# =====================================================
print("\n" + "=" * 60)
print("Test 4: SOFTWARE_ADMIN CurrentUser")
print("=" * 60)

software_admin = CurrentUser(
    user_id=1,
    username="software_admin",
    is_active=True,
    scopes=[UserScope(
        role_code="SOFTWARE_ADMIN",
        org_unit_id=0,
        org_unit_type="ADMINISTRATION"
    )]
)

software_admin.allowed_unit_ids = resolve_user_scope(software_admin)

print(f"✓ Username: {software_admin.username}")
print(f"✓ User ID: {software_admin.user_id}")
print(f"✓ Scopes: {len(software_admin.scopes)} scope(s)")
print(f"✓ Allowed Unit IDs: {len(software_admin.allowed_unit_ids)} units")
print(f"  Expected: All {len(full_tree)} units")
print(f"  Result: {'✅ PASS' if len(software_admin.allowed_unit_ids) == len(full_tree) else '❌ FAIL'}")

# =====================================================
# Test 5: Verify field exists and is accessible
# =====================================================
print("\n" + "=" * 60)
print("Test 5: CurrentUser Model Fields")
print("=" * 60)

print(f"✓ CurrentUser has 'user_id': {hasattr(section_user, 'user_id')}")
print(f"✓ CurrentUser has 'username': {hasattr(section_user, 'username')}")
print(f"✓ CurrentUser has 'is_active': {hasattr(section_user, 'is_active')}")
print(f"✓ CurrentUser has 'scopes': {hasattr(section_user, 'scopes')}")
print(f"✓ CurrentUser has 'allowed_unit_ids': {hasattr(section_user, 'allowed_unit_ids')}")
print(f"  Result: {'✅ PASS' if hasattr(section_user, 'allowed_unit_ids') else '❌ FAIL'}")

# =====================================================
# Test 6: Type checking
# =====================================================
print("\n" + "=" * 60)
print("Test 6: Field Type Validation")
print("=" * 60)

print(f"✓ allowed_unit_ids is a set: {isinstance(section_user.allowed_unit_ids, set)}")
print(f"✓ allowed_unit_ids contains ints: {all(isinstance(x, int) for x in section_user.allowed_unit_ids)}")
print(f"  Result: {'✅ PASS' if isinstance(section_user.allowed_unit_ids, set) else '❌ FAIL'}")

# =====================================================
# Test 7: Error handling - Invalid configuration
# =====================================================
print("\n" + "=" * 60)
print("Test 7: Invalid Configuration (Should Fail)")
print("=" * 60)

invalid_user = CurrentUser(
    user_id=99,
    username="invalid_user",
    is_active=True,
    scopes=[]  # No scopes
)

try:
    invalid_user.allowed_unit_ids = resolve_user_scope(invalid_user)
    print("❌ FAIL: Should have raised ValueError for 0 scopes")
except ValueError as e:
    print(f"✅ PASS: Correctly raised error: {str(e)[:80]}...")

print("\n" + "=" * 60)
print("✓ All tests completed - CurrentUser upgrade successful")
print("=" * 60)
