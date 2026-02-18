"""
Create test inbox data for three admin users.
Creates 5 subcases for each user at their respective workflow levels.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.database import get_connection
from datetime import datetime

def get_user_org_units():
    """Get org unit assignments for the three test users."""
    conn = get_connection()
    cursor = conn.cursor()
    
    users = {}
    
    # Get user org unit assignments
    query = """
    SELECT u.UserID, u.Username, urs.OrgUnitID, urs.OrgUnitType
    FROM APP_Users u
    JOIN APP_UserRoleScope urs ON u.UserID = urs.UserID
    WHERE u.Username IN ('adm_1_admin', 'dept_15_admin', 'sec_100_admin')
    """
    
    cursor.execute(query)
    for row in cursor.fetchall():
        user_id, username, org_unit_id, org_unit_type = row
        users[username] = {
            'user_id': user_id,
            'org_unit_id': org_unit_id,
            'org_unit_type': org_unit_type
        }
    
    cursor.close()
    conn.close()
    
    return users

def create_test_subcases():
    """Create test subcases for each user."""
    users = get_user_org_units()
    
    if not users:
        print("❌ No users found. Make sure users exist in database.")
        return
    
    print("\n" + "="*80)
    print("CREATING TEST INBOX DATA")
    print("="*80)
    
    for username, info in users.items():
        print(f"\nUser: {username}")
        print(f"  UserID: {info['user_id']}")
        print(f"  OrgUnitID: {info['org_unit_id']}")
        print(f"  OrgUnitType: {info['org_unit_type']}")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Get some incident IDs to link subcases to
    cursor.execute("SELECT TOP 15 IncidentRequestCaseID FROM APP_IncidentCase ORDER BY CreatedAt DESC")
    incident_ids = [row[0] for row in cursor.fetchall()]
    
    if len(incident_ids) < 15:
        print(f"\n❌ Not enough incidents found. Need 15, found {len(incident_ids)}")
        cursor.close()
        conn.close()
        return
    
    print(f"\nUsing {len(incident_ids)} incident IDs for test data")
    
    print(f"\nUsing {len(incident_ids)} incident IDs for test data")
    
    incident_idx = 0
    
    # Section Admin (sec_100_admin) - needs SUBMITTED_TO_SECTION subcases
    if 'sec_100_admin' in users:
        user = users['sec_100_admin']
        print(f"\n\nCreating 5 subcases for sec_100_admin (target: {user['org_unit_id']})...")
        for i in range(5):
            cursor.execute("""
                INSERT INTO APP_AdministrativeSubcase (
                    CaseType, 
                    IncidentRequestCaseID,
                    TargetOrgUnitID,
                    Status,
                    CreatedAt,
                    CreatedByUserID
                )
                VALUES ('INCIDENT', ?, ?, 'SUBMITTED_TO_SECTION', GETDATE(), ?)
            """, incident_ids[incident_idx], user['org_unit_id'], user['user_id'])
            incident_idx += 1
            print(f"  ✓ Created subcase {i+1}")
        conn.commit()
    
    # Department Admin (dept_15_admin) - needs SECTION_ACCEPTED_PENDING_DEPT subcases
    if 'dept_15_admin' in users:
        user = users['dept_15_admin']
        print(f"\n\nCreating 5 subcases for dept_15_admin (target: {user['org_unit_id']})...")
        for i in range(5):
            cursor.execute("""
                INSERT INTO APP_AdministrativeSubcase (
                    CaseType,
                    IncidentRequestCaseID,
                    TargetOrgUnitID,
                    Status,
                    CreatedAt,
                    CreatedByUserID
                )
                VALUES ('INCIDENT', ?, ?, 'SECTION_ACCEPTED_PENDING_DEPT', GETDATE(), ?)
            """, incident_ids[incident_idx], user['org_unit_id'], user['user_id'])
            incident_idx += 1
            print(f"  ✓ Created subcase {i+1}")
        conn.commit()
    
    # Administration Admin (adm_1_admin) - needs DEPT_ACCEPTED_PENDING_ADMIN subcases
    if 'adm_1_admin' in users:
        user = users['adm_1_admin']
        print(f"\n\nCreating 5 subcases for adm_1_admin (target: {user['org_unit_id']})...")
        for i in range(5):
            cursor.execute("""
                INSERT INTO APP_AdministrativeSubcase (
                    CaseType,
                    IncidentRequestCaseID,
                    TargetOrgUnitID,
                    Status,
                    CreatedAt,
                    CreatedByUserID
                )
                VALUES ('INCIDENT', ?, ?, 'DEPT_ACCEPTED_PENDING_ADMIN', GETDATE(), ?)
            """, incident_ids[incident_idx], user['org_unit_id'], user['user_id'])
            incident_idx += 1
            print(f"  ✓ Created subcase {i+1}")
        conn.commit()
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*80)
    print("✅ TEST DATA CREATION COMPLETE")
    print("="*80)
    print("\nYou can now test the inbox endpoints:")
    print("  - sec_100_admin should see 5 items")
    print("  - dept_15_admin should see 5 items")
    print("  - adm_1_admin should see 5 items")
    print()

if __name__ == '__main__':
    try:
        create_test_subcases()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
