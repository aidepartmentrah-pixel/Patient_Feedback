"""
Test script to verify inbox endpoint returns correct allowedActions for each role.
Tests the three admin users we created test data for.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api_v2.services import inbox_service
from backend.api.schemas.auth_models import CurrentUser, UserScope

def create_mock_user(username, role_code, org_unit_id, org_unit_type, allowed_unit_ids):
    """Create a mock user for testing."""
    user = CurrentUser(
        user_id=1,
        username=username,
        is_active=True,
        scopes=[UserScope(
            role_code=role_code,
            org_unit_id=org_unit_id,
            org_unit_type=org_unit_type
        )]
    )
    user.allowed_unit_ids = allowed_unit_ids
    return user

def test_role(username, role_code, org_unit_id, org_unit_type, allowed_unit_ids):
    """Test inbox for a specific role."""
    print(f"\n{'='*80}")
    print(f"Testing: {username} ({role_code})")
    print(f"{'='*80}")
    
    user = create_mock_user(username, role_code, org_unit_id, org_unit_type, allowed_unit_ids)
    
    try:
        inbox = inbox_service.get_inbox(user)
        print(f"✅ Inbox items: {len(inbox)}")
        
        if inbox:
            # Show first item details
            item = inbox[0]
            print(f"\nFirst item:")
            print(f"  subcase_id: {item.get('subcase_id')}")
            print(f"  status: {item.get('status')}")
            print(f"  allowed_actions: {item.get('allowed_actions')}")
            
            # Verify allowed_actions format
            actions = item.get('allowed_actions', [])
            if not actions:
                print("  ⚠️ WARNING: No actions returned!")
            else:
                # Check if actions are lowercase
                uppercase_actions = [a for a in actions if a != a.lower()]
                if uppercase_actions:
                    print(f"  ❌ ERROR: Found uppercase actions: {uppercase_actions}")
                else:
                    print(f"  ✅ All actions are lowercase")
        else:
            print("  ℹ️ No items in inbox")
            
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("\n" + "="*80)
    print("INBOX ALLOWED ACTIONS TEST")
    print("="*80)
    
    # Get all org units for full-access roles
    from backend.api.services import org_tree_service
    full_tree = org_tree_service.get_full_tree()
    all_unit_ids = {unit["UniqueID"] for unit in full_tree}
    
    # Test SECTION_ADMIN (should see submit_response and reject)
    test_role(
        username="sec_100_admin",
        role_code="SECTION_ADMIN",
        org_unit_id=100,
        org_unit_type="SECTION",
        allowed_unit_ids={100}  # Only their section
    )
    
    # Test DEPARTMENT_ADMIN (should see accept and reject)
    test_role(
        username="dept_15_admin",
        role_code="DEPARTMENT_ADMIN",
        org_unit_id=15,
        org_unit_type="DEPARTMENT",
        allowed_unit_ids=org_tree_service.get_descendants(15)  # Department + sections
    )
    
    # Test ADMINISTRATION_ADMIN (should see accept and reject)
    test_role(
        username="adm_1_admin",
        role_code="ADMINISTRATION_ADMIN",
        org_unit_id=1,
        org_unit_type="ADMINISTRATION",
        allowed_unit_ids=org_tree_service.get_descendants(1)  # Admin + all children
    )
    
    # Test SOFTWARE_ADMIN (should see accept and reject for all items)
    test_role(
        username="software_admin",
        role_code="SOFTWARE_ADMIN",
        org_unit_id=0,
        org_unit_type="ADMINISTRATION",
        allowed_unit_ids=all_unit_ids  # All units
    )
    
    # Test WORKER (should see view only)
    test_role(
        username="worker",
        role_code="WORKER",
        org_unit_id=10,
        org_unit_type="COMPLAINT",
        allowed_unit_ids=all_unit_ids  # All units (full access as per recent fix)
    )
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE")
    print("="*80)
