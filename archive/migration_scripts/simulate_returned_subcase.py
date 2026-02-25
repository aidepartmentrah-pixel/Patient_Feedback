"""
Simulation: Create a RETURNED_TO_SECTION_FOR_REVISION subcase and verify 
it appears in the section admin's inbox through the full chain.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.dirname(__file__))

from core.database import get_connection


def simulate_returned_subcase():
    """Create a temporary RETURNED subcase, test inbox, then clean up."""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Find a valid TargetOrgUnitID from existing SUBMITTED_TO_SECTION subcases
    cursor.execute("""
        SELECT TOP 1 SubcaseID, TargetOrgUnitID, CreatedByUserID
        FROM dbo.APP_AdministrativeSubcase 
        WHERE Status = 'SUBMITTED_TO_SECTION'
    """)
    ref = cursor.fetchone()
    if not ref:
        print("No SUBMITTED_TO_SECTION subcases to reference")
        return
    
    ref_subcase_id = ref[0]
    target_org_unit_id = ref[1]
    created_by = ref[2]
    
    print(f"Reference subcase: ID={ref_subcase_id}, OrgUnit={target_org_unit_id}, CreatedBy={created_by}")
    
    # Update that subcase to RETURNED_TO_SECTION_FOR_REVISION temporarily
    print(f"\nTemporarily setting SubcaseID={ref_subcase_id} to RETURNED_TO_SECTION_FOR_REVISION...")
    cursor.execute("""
        UPDATE dbo.APP_AdministrativeSubcase 
        SET Status = 'RETURNED_TO_SECTION_FOR_REVISION'
        WHERE SubcaseID = ?
    """, (ref_subcase_id,))
    conn.commit()
    
    try:
        # TEST 1: DB layer query
        print("\n--- TEST 1: DB Layer ---")
        from api_v2.db_layer import administrative_subcase_db
        
        pending = administrative_subcase_db.get_subcases_pending_for_section()
        returned_items = [s for s in pending if s.get('status') == 'RETURNED_TO_SECTION_FOR_REVISION']
        print(f"  get_subcases_pending_for_section() returned {len(pending)} total, {len(returned_items)} with RETURNED status")
        
        if returned_items:
            print(f"  PASS: RETURNED subcases found in DB layer ✓")
            for item in returned_items:
                print(f"    SubcaseID={item['subcase_id']}, TargetOrgUnitID={item['target_org_unit_id']}, Status={item['status']}")
        else:
            print(f"  FAIL: No RETURNED subcases found in DB layer ✗")
        
        # TEST 2: Service layer with a real user context
        print("\n--- TEST 2: Scope filter simulation ---")
        # Find section admin users who have access to this org unit
        cursor2 = conn.cursor()
        cursor2.execute("""
            SELECT TOP 1 ur.UserID, ur.RoleCode, ur.OrgUnitID, u.Username
            FROM dbo.APP_UserOrgUnitRole ur
            JOIN dbo.APP_User u ON ur.UserID = u.UserID
            WHERE ur.RoleCode = 'SECTION_ADMIN' AND ur.OrgUnitID = ?
        """, (target_org_unit_id,))
        admin_row = cursor2.fetchone()
        cursor2.close()
        
        if admin_row:
            print(f"  Section admin found: UserID={admin_row[0]}, Username={admin_row[3]}, OrgUnitID={admin_row[2]}")
            
            # Simulate the scope filter check
            print(f"  Target org unit of returned subcase: {target_org_unit_id}")
            print(f"  Section admin's direct org unit: {admin_row[2]}")
            print(f"  Match: {target_org_unit_id == admin_row[2]}")
            
            # Check what allowed_unit_ids this user would get from scope engine
            cursor3 = conn.cursor()
            cursor3.execute("""
                SELECT OrgUnitID FROM dbo.APP_UserOrgUnitRole WHERE UserID = ?
            """, (admin_row[0],))
            user_units = [r[0] for r in cursor3.fetchall()]
            cursor3.close()
            
            print(f"  User's org unit assignments: {user_units}")
            print(f"  Target org unit {target_org_unit_id} in user assignments: {target_org_unit_id in user_units}")
            
            # Now check the scope engine (allowed_unit_ids)
            print("\n--- TEST 3: Phase 2.5 Scope Engine ---")
            try:
                from api.dependencies.user_context import _build_allowed_unit_ids
                allowed = _build_allowed_unit_ids(admin_row[2], admin_row[1])
                print(f"  _build_allowed_unit_ids({admin_row[2]}, '{admin_row[1]}'): {allowed}")
                print(f"  Target org unit {target_org_unit_id} in allowed_unit_ids: {target_org_unit_id in allowed}")
            except ImportError:
                print("  Could not import scope engine, checking alternative...")
                try:
                    from api_v2.scope_engine import build_allowed_unit_ids
                    allowed = build_allowed_unit_ids(admin_row[2], admin_row[1])
                    print(f"  build_allowed_unit_ids: {allowed}")
                    print(f"  Target org unit {target_org_unit_id} in allowed_unit_ids: {target_org_unit_id in allowed}")
                except ImportError:
                    print("  Could not import scope engine from either path")
                    # Manual check: just see if the org unit is in the tree
                    cursor4 = conn.cursor()
                    cursor4.execute("""
                        SELECT UniqueID, Name, ParentUniqueID, Type 
                        FROM dbo.AdminsrationUnit 
                        WHERE UniqueID = ?
                    """, (target_org_unit_id,))
                    org = cursor4.fetchone()
                    cursor4.close()
                    if org:
                        print(f"  Org Unit: ID={org[0]}, Name={org[1]}, Parent={org[2]}, Type={org[3]}")
                    
        else:
            print(f"  No section admin assigned to org unit {target_org_unit_id}")
            print("  This could mean: no section admin can see subcases for this org unit!")
            
            # Check what section admins exist and their org units
            cursor4 = conn.cursor()
            cursor4.execute("""
                SELECT TOP 5 ur.UserID, ur.OrgUnitID, u.Username
                FROM dbo.APP_UserOrgUnitRole ur
                JOIN dbo.APP_User u ON ur.UserID = u.UserID
                WHERE ur.RoleCode = 'SECTION_ADMIN'
            """)
            admins = cursor4.fetchall()
            cursor4.close()
            print(f"  Existing section admins:")
            for a in admins:
                print(f"    UserID={a[0]}, OrgUnitID={a[1]}, Username={a[2]}")
        
        # TEST 4: Test the allowed_actions for RETURNED status
        print("\n--- TEST 4: Allowed actions for RETURNED_TO_SECTION_FOR_REVISION ---")
        from api_v2.services.inbox_service import _compute_allowed_actions
        
        mock_subcase = {'status': 'RETURNED_TO_SECTION_FOR_REVISION', 'subcase_id': ref_subcase_id}
        
        class MockScope:
            def __init__(self, role):
                self.role_code = role
        
        class MockUser:
            scopes = [MockScope('SECTION_ADMIN')]
        
        actions = _compute_allowed_actions(mock_subcase, MockUser())
        print(f"  SECTION_ADMIN + RETURNED_TO_SECTION_FOR_REVISION → {actions}")
        expected = ["view", "submit_response", "reject"]
        print(f"  Expected: {expected}")
        print(f"  Match: {actions == expected} ✓" if actions == expected else f"  MISMATCH ✗")
        
        # TEST 5: reject_responsibility bug confirmation
        print("\n--- TEST 5: reject_responsibility status assertion ---")
        from api_v2.services.case_response_service import _assert_status
        
        returned_subcase = {'status': 'RETURNED_TO_SECTION_FOR_REVISION', 'subcase_id': ref_subcase_id}
        try:
            _assert_status(returned_subcase, ['SUBMITTED_TO_SECTION'])
            print("  reject_responsibility would PASS ✓ (unexpected)")
        except Exception as e:
            print(f"  reject_responsibility would FAIL: {e}")
            print("  BUG CONFIRMED: Section admin cannot REJECT from RETURNED status ✗")
    
    finally:
        # RESTORE original status
        print(f"\nRestoring SubcaseID={ref_subcase_id} to SUBMITTED_TO_SECTION...")
        cursor.execute("""
            UPDATE dbo.APP_AdministrativeSubcase 
            SET Status = 'SUBMITTED_TO_SECTION'
            WHERE SubcaseID = ?
        """, (ref_subcase_id,))
        conn.commit()
        print("Restored ✓")
        cursor.close()
        conn.close()


if __name__ == "__main__":
    simulate_returned_subcase()
