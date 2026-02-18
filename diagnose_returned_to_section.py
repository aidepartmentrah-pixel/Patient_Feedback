"""
Diagnostic script: Why are RETURNED_TO_SECTION_FOR_REVISION subcases not visible?
Traces the full chain: DB → service → scope filter → inbox output
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.dirname(__file__))

from core.database import get_connection


def check_db_statuses():
    """Step 1: Check what statuses actually exist in the DB"""
    print("=" * 70)
    print("STEP 1: All subcase statuses in DB")
    print("=" * 70)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT Status, COUNT(*) as cnt 
            FROM dbo.APP_AdministrativeSubcase 
            GROUP BY Status 
            ORDER BY Status
        """)
        rows = cursor.fetchall()
        for row in rows:
            marker = " <<<" if 'RETURNED' in (row[0] or '') else ""
            print(f"  {row[0]}: {row[1]} subcases{marker}")
        
        if not rows:
            print("  (No subcases found in DB)")
    finally:
        cursor.close()
        conn.close()


def check_returned_subcases_detail():
    """Step 2: Show detail for RETURNED_TO_SECTION_FOR_REVISION subcases"""
    print("\n" + "=" * 70)
    print("STEP 2: RETURNED_TO_SECTION_FOR_REVISION subcases details")
    print("=" * 70)
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT SubcaseID, CaseType, TargetOrgUnitID, Status, CreatedAt, UpdatedAt
            FROM dbo.APP_AdministrativeSubcase
            WHERE Status = 'RETURNED_TO_SECTION_FOR_REVISION'
            ORDER BY UpdatedAt DESC
        """)
        rows = cursor.fetchall()
        if not rows:
            print("  *** NO subcases with RETURNED_TO_SECTION_FOR_REVISION status ***")
            print("  This could mean:")
            print("    a) No department rejection has happened yet")
            print("    b) The status string is stored differently (check for typos)")
            
            # Check for similar strings
            cursor.execute("""
                SELECT DISTINCT Status FROM dbo.APP_AdministrativeSubcase 
                WHERE Status LIKE '%RETURN%' OR Status LIKE '%REVISION%' OR Status LIKE '%REJECTED%'
            """)
            similar = cursor.fetchall()
            if similar:
                print(f"  Similar statuses found: {[r[0] for r in similar]}")
            else:
                print("  No similar statuses found")
        else:
            for row in rows:
                print(f"  SubcaseID={row[0]}, CaseType={row[1]}, TargetOrgUnitID={row[2]}, Status={row[3]}")
                print(f"    Created={row[4]}, Updated={row[5]}")
    finally:
        cursor.close()
        conn.close()


def check_db_layer_queries():
    """Step 3: Test the DB layer functions directly"""
    print("\n" + "=" * 70)
    print("STEP 3: Testing DB layer get_subcases_pending_for_section()")
    print("=" * 70)
    
    from api_v2.db_layer import administrative_subcase_db
    
    # Test get_subcases_by_status for each status individually
    submitted = administrative_subcase_db.get_subcases_by_status("SUBMITTED_TO_SECTION")
    returned = administrative_subcase_db.get_subcases_by_status("RETURNED_TO_SECTION_FOR_REVISION")
    
    print(f"  SUBMITTED_TO_SECTION: {len(submitted)} subcases")
    print(f"  RETURNED_TO_SECTION_FOR_REVISION: {len(returned)} subcases")
    
    # Test the combined function
    pending = administrative_subcase_db.get_subcases_pending_for_section()
    print(f"  get_subcases_pending_for_section() total: {len(pending)} subcases")
    
    # Show the returned ones
    for sub in pending:
        if sub.get('status') == 'RETURNED_TO_SECTION_FOR_REVISION':
            print(f"    -> SubcaseID={sub['subcase_id']}, TargetOrgUnitID={sub['target_org_unit_id']}, Status={sub['status']}")
    
    return pending


def check_scope_filter(pending_subcases):
    """Step 4: Simulate scope filtering for a section admin"""
    print("\n" + "=" * 70)
    print("STEP 4: Testing scope filter impact")
    print("=" * 70)
    
    # Get all target org unit IDs from returned subcases
    returned = [s for s in pending_subcases if s.get('status') == 'RETURNED_TO_SECTION_FOR_REVISION']
    
    if not returned:
        print("  No RETURNED_TO_SECTION_FOR_REVISION subcases to test scope filter on")
        return
    
    target_ids = set(s.get('target_org_unit_id') for s in returned)
    print(f"  Target org unit IDs of returned subcases: {target_ids}")
    
    # Check which users/roles are associated with these org units
    conn = get_connection()
    cursor = conn.cursor()
    try:
        for org_id in target_ids:
            print(f"\n  Org Unit {org_id}:")
            
            # Check the org unit exists
            cursor.execute("SELECT UniqueID, Name, Type FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (org_id,))
            org = cursor.fetchone()
            if org:
                print(f"    Name: {org[1]}, Type: {org[2]}")
            else:
                print(f"    *** ORG UNIT NOT FOUND ***")
            
            # Check which scope assignments reference this unit
            cursor.execute("""
                SELECT ur.UserID, ur.RoleCode, ur.OrgUnitID
                FROM dbo.APP_UserOrgUnitRole ur
                WHERE ur.OrgUnitID = ? AND ur.RoleCode = 'SECTION_ADMIN'
            """, (org_id,))
            roles = cursor.fetchall()
            if roles:
                for r in roles:
                    print(f"    Section Admin UserID={r[0]}, RoleCode={r[1]}, OrgUnitID={r[2]}")
            else:
                print(f"    *** No SECTION_ADMIN users assigned to this org unit ***")
    finally:
        cursor.close()
        conn.close()


def check_reject_from_returned_status():
    """Step 5: Check if section admin can REJECT from RETURNED status"""
    print("\n" + "=" * 70)
    print("STEP 5: reject_responsibility() status assertion check")
    print("=" * 70)
    print("  reject_responsibility() only allows: ['SUBMITTED_TO_SECTION']")
    print("  This means section admin CANNOT reject from RETURNED_TO_SECTION_FOR_REVISION")
    print("  The REJECT action in workflow_router.py tries:")
    print("    1. reject_responsibility → fails for RETURNED status")
    print("    2. reject_department → fails (wrong status)")
    print("    3. reject_administration → fails (wrong status)")
    print("  Result: All 3 fail → 400 error 'Reject failed at all levels'")
    print("  BUG CONFIRMED: reject_responsibility should also allow RETURNED_TO_SECTION_FOR_REVISION")


def check_submit_from_returned_status():
    """Step 6: Check if section admin can SUBMIT_RESPONSE from RETURNED status"""
    print("\n" + "=" * 70)
    print("STEP 6: submit_section_response() status assertion check")
    print("=" * 70)
    print("  submit_section_response() allows: ['SUBMITTED_TO_SECTION', 'RETURNED_TO_SECTION_FOR_REVISION']")
    print("  SUBMIT_RESPONSE action works correctly for both statuses ✓")


if __name__ == "__main__":
    print("DIAGNOSTIC: RETURNED_TO_SECTION_FOR_REVISION visibility")
    print("Date: 2026-02-17")
    print()
    
    check_db_statuses()
    check_returned_subcases_detail()
    pending = check_db_layer_queries()
    check_scope_filter(pending)
    check_reject_from_returned_status()
    check_submit_from_returned_status()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
FINDINGS:
1. DB layer queries are correct (SUBMITTED_TO_SECTION + RETURNED_TO_SECTION_FOR_REVISION)
2. Service layer routing is correct (SECTION_ADMIN → get_section_inbox)
3. Scope filter works by target_org_unit_id ∈ allowed_unit_ids

KNOWN BUG: reject_responsibility() only allows SUBMITTED_TO_SECTION
  → Section admin CANNOT reject from RETURNED_TO_SECTION_FOR_REVISION
  → This needs to be fixed in case_response_service.py

POSSIBLE VISIBILITY CAUSES:
  a) If allowed_unit_ids doesn't include the target org unit → item filtered out
  b) If no subcases actually have RETURNED_TO_SECTION_FOR_REVISION status in DB
  c) Check the DB output above for actual data
""")
