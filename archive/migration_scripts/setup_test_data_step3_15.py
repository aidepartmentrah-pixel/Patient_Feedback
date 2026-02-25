"""
STEP 3.15 — Test Data Setup Script

Creates test data for insight_service.py testing:
- Proper organizational hierarchy (Administration -> Department -> Section)
- Test subcases (open and closed)
- Test action items (some overdue)

This ensures we can fully test hierarchy-based scoping and all insight functions.
"""

import sys
import os
from datetime import datetime, timedelta

# Force UTF-8 encoding for emoji support
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection


def get_db_cursor():
    """Get database cursor"""
    conn = get_connection()
    cursor = conn.cursor()
    return conn, cursor


def check_existing_data():
    """Check what organizational data already exists"""
    print("\n" + "="*80)
    print("CHECKING EXISTING DATA")
    print("="*80)
    
    conn, cursor = get_db_cursor()
    try:
        # Check org unit types
        cursor.execute("""
            SELECT Type, COUNT(*) as Count
            FROM dbo.AdminsrationUnit
            GROUP BY Type
            ORDER BY Type
        """)
        
        print("\n[ORG UNITS BY TYPE]")
        for row in cursor.fetchall():
            type_name = {1: "Administration", 2: "Department", 3: "Section"}.get(row.Type, f"Unknown({row.Type})")
            print(f"  Type {row.Type} ({type_name}): {row.Count} units")
        
        # Check hierarchical relationships
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT CASE WHEN ParentID IS NULL THEN UniqueID END) as RootNodes,
                COUNT(DISTINCT CASE WHEN ParentID IS NOT NULL THEN UniqueID END) as ChildNodes,
                COUNT(DISTINCT ParentID) as ParentsWithChildren
            FROM dbo.AdminsrationUnit
        """)
        
        row = cursor.fetchone()
        print(f"\n[HIERARCHY STATUS]")
        print(f"  Root nodes (no parent): {row.RootNodes}")
        print(f"  Child nodes (has parent): {row.ChildNodes}")
        print(f"  Parents with children: {row.ParentsWithChildren}")
        
        # Check if we have a proper hierarchy
        cursor.execute("""
            SELECT TOP 1
                admin.UniqueID as AdminID,
                admin.Name as AdminName,
                dept.UniqueID as DeptID,
                dept.Name as DeptName,
                sec.UniqueID as SecID,
                sec.Name as SecName
            FROM dbo.AdminsrationUnit admin
            INNER JOIN dbo.AdminsrationUnit dept ON dept.ParentID = admin.UniqueID
            INNER JOIN dbo.AdminsrationUnit sec ON sec.ParentID = dept.UniqueID
            WHERE admin.Type = 1 AND dept.Type = 2 AND sec.Type = 3
        """)
        
        hierarchy = cursor.fetchone()
        if hierarchy:
            print(f"\n[FOUND PROPER HIERARCHY]")
            print(f"  ✅ Administration: {hierarchy.AdminName} (ID={hierarchy.AdminID})")
            print(f"  ✅ Department: {hierarchy.DeptName} (ID={hierarchy.DeptID})")
            print(f"  ✅ Section: {hierarchy.SecName} (ID={hierarchy.SecID})")
            return {
                'has_hierarchy': True,
                'admin_id': hierarchy.AdminID,
                'dept_id': hierarchy.DeptID,
                'section_id': hierarchy.SecID
            }
        else:
            print(f"\n[NO PROPER HIERARCHY FOUND]")
            print(f"  ⚠️  Need to create test hierarchy")
            return {'has_hierarchy': False}
        
    finally:
        cursor.close()
        conn.close()


def create_test_hierarchy():
    """Create a test organizational hierarchy if needed"""
    print("\n" + "="*80)
    print("CREATING TEST HIERARCHY")
    print("="*80)
    
    conn, cursor = get_db_cursor()
    try:
        # Create Administration
        cursor.execute("""
            INSERT INTO dbo.AdminsrationUnit (ParentID, Type, Name, Frozen)
            VALUES (NULL, 1, 'TEST ADMINISTRATION', 0)
        """)
        cursor.execute("SELECT @@IDENTITY")
        admin_id = int(cursor.fetchone()[0])
        print(f"\n✅ Created Administration: TEST ADMINISTRATION (ID={admin_id})")
        
        # Create Department under Administration
        cursor.execute("""
            INSERT INTO dbo.AdminsrationUnit (ParentID, Type, Name, Frozen)
            VALUES (?, 2, 'TEST DEPARTMENT', 0)
        """, (admin_id,))
        cursor.execute("SELECT @@IDENTITY")
        dept_id = int(cursor.fetchone()[0])
        print(f"✅ Created Department: TEST DEPARTMENT (ID={dept_id})")
        
        # Create 3 Sections under Department
        section_ids = []
        for i in range(1, 4):
            cursor.execute("""
                INSERT INTO dbo.AdminsrationUnit (ParentID, Type, Name, Frozen)
                VALUES (?, 3, ?, 0)
            """, (dept_id, f'TEST SECTION {i}'))
            cursor.execute("SELECT @@IDENTITY")
            sec_id = int(cursor.fetchone()[0])
            section_ids.append(sec_id)
            print(f"✅ Created Section: TEST SECTION {i} (ID={sec_id})")
        
        conn.commit()
        
        print(f"\n✅ Test hierarchy created successfully!")
        return {
            'has_hierarchy': True,
            'admin_id': admin_id,
            'dept_id': dept_id,
            'section_ids': section_ids
        }
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error creating hierarchy: {str(e)}")
        raise
    finally:
        cursor.close()
        conn.close()


def create_test_subcases(hierarchy_info):
    """Create test subcases for different sections"""
    print("\n" + "="*80)
    print("CREATING TEST SUBCASES")
    print("="*80)
    
    from api_v2.db_layer import administrative_subcase_db
    
    section_ids = hierarchy_info.get('section_ids', [hierarchy_info.get('section_id')])
    
    subcase_ids = []
    
    # First, create a minimal test incident case for linking
    conn, cursor = get_db_cursor()
    try:
        # Use minimal required fields
        cursor.execute("""
            INSERT INTO dbo.APP_IncidentCase 
            (ComplaintText, FeedbackRecievedDate, IssuingOrgUnitID, CreatedByUserID, CreatedAt)
            VALUES (?, GETDATE(), ?, 1, GETDATE())
        """, ('TEST INCIDENT FOR INSIGHT SERVICE', section_ids[0] if section_ids else 1))
        cursor.execute("SELECT @@IDENTITY")
        incident_id = int(cursor.fetchone()[0])
        conn.commit()
        print(f"✅ Created test incident {incident_id}")
    except Exception as e:
        conn.rollback()
        print(f"❌ Error creating incident: {str(e)}")
        # If we can't create an incident, try to use an existing one
        try:
            cursor.execute("SELECT TOP 1 IncidentRequestCaseID FROM dbo.APP_IncidentCase")
            row = cursor.fetchone()
            if row:
                incident_id = row[0]
                print(f"⚠️  Using existing incident {incident_id} for testing")
            else:
                raise Exception("No incidents available for testing")
        except:
            raise
    finally:
        cursor.close()
        conn.close()
    
    # Create 2 open subcases for first section using db_layer
    for i in range(2):
        subcase_id = administrative_subcase_db.create_subcase(
            case_type='INCIDENT_RESPONSE',
            incident_id=incident_id,
            seasonal_report_id=None,
            target_org_unit_id=section_ids[0],
            created_by_user_id=1,
            initial_status='SUBMITTED_TO_SECTION'
        )
        if subcase_id:
            subcase_ids.append(subcase_id)
            print(f"✅ Created open subcase {subcase_id} for Section {section_ids[0]}")
    
    # Create 1 open subcase for second section (if exists)
    if len(section_ids) > 1:
        subcase_id = administrative_subcase_db.create_subcase(
            case_type='INCIDENT_RESPONSE',
            incident_id=incident_id,
            seasonal_report_id=None,
            target_org_unit_id=section_ids[1],
            created_by_user_id=1,
            initial_status='UNDER_REVIEW'
        )
        if subcase_id:
            subcase_ids.append(subcase_id)
            print(f"✅ Created open subcase {subcase_id} for Section {section_ids[1]}")
    
    # Create 1 closed subcase for third section (if exists)
    if len(section_ids) > 2:
        closed_id = administrative_subcase_db.create_subcase(
            case_type='INCIDENT_RESPONSE',
            incident_id=incident_id,
            seasonal_report_id=None,
            target_org_unit_id=section_ids[2],
            created_by_user_id=1,
            initial_status='CLOSED'
        )
        if closed_id:
            print(f"✅ Created closed subcase {closed_id} for Section {section_ids[2]} (should be filtered out)")
    
    print(f"\n✅ Created {len(subcase_ids)} open test subcases")
    return subcase_ids


def create_test_action_items(subcase_ids):
    """Create test action items (some overdue)"""
    print("\n" + "="*80)
    print("CREATING TEST ACTION ITEMS")
    print("="*80)
    
    from api_v2.db_layer import action_item_subcase_db
    from datetime import date
    
    # Create overdue action item for first subcase
    yesterday = date.today() - timedelta(days=1)
    overdue_id = action_item_subcase_db.create_action_item(
        subcase_id=subcase_ids[0],
        title='OVERDUE Action Item',
        description='This is overdue and should appear in bottlenecks',
        due_date=yesterday,
        created_by_user_id=1,
        initial_status='IN_PROGRESS'  # Valid status code
    )
    if overdue_id:
        print(f"✅ Created OVERDUE action item {overdue_id} for subcase {subcase_ids[0]}")
    
    # Create on-time action item for first subcase
    tomorrow = date.today() + timedelta(days=1)
    ontime_id = action_item_subcase_db.create_action_item(
        subcase_id=subcase_ids[0],
        title='On-time Action Item',
        description='This is not overdue',
        due_date=tomorrow,
        created_by_user_id=1,
        initial_status='IN_PROGRESS'  # Valid status code
    )
    if ontime_id:
        print(f"✅ Created on-time action item {ontime_id} for subcase {subcase_ids[0]}")
    
    # Create action item for second subcase if exists
    if len(subcase_ids) > 1:
        normal_id = action_item_subcase_db.create_action_item(
            subcase_id=subcase_ids[1],
            title='Normal Action Item',
            description='This has no due date',
            due_date=None,
            created_by_user_id=1,
            initial_status='DRAFT'  # Valid status code
        )
        if normal_id:
            print(f"✅ Created normal action item {normal_id} for subcase {subcase_ids[1]}")
    
    print(f"\n✅ Test action items created successfully")


def verify_test_data():
    """Verify the test data was created correctly"""
    print("\n" + "="*80)
    print("VERIFYING TEST DATA")
    print("="*80)
    
    conn, cursor = get_db_cursor()
    try:
        # Check subcases
        cursor.execute("""
            SELECT 
                Status,
                COUNT(*) as Count
            FROM dbo.APP_AdministrativeSubcase
            GROUP BY Status
        """)
        
        print("\n[SUBCASES BY STATUS]")
        for row in cursor.fetchall():
            print(f"  {row.Status}: {row.Count}")
        
        # Check action items
        cursor.execute("""
            SELECT 
                COUNT(*) as Total,
                COUNT(CASE WHEN DueDate < GETDATE() AND Status NOT IN ('COMPLETED', 'CLOSED') THEN 1 END) as Overdue
            FROM dbo.APP_SubcaseActionItem
        """)
        
        row = cursor.fetchone()
        print(f"\n[ACTION ITEMS]")
        print(f"  Total: {row.Total}")
        print(f"  Overdue: {row.Overdue}")
        
        # Check hierarchy
        cursor.execute("""
            SELECT 
                admin.Name as Admin,
                dept.Name as Dept,
                sec.Name as Section,
                COUNT(sc.SubcaseID) as Subcases
            FROM dbo.AdminsrationUnit admin
            LEFT JOIN dbo.AdminsrationUnit dept ON dept.ParentID = admin.UniqueID
            LEFT JOIN dbo.AdminsrationUnit sec ON sec.ParentID = dept.UniqueID
            LEFT JOIN dbo.APP_AdministrativeSubcase sc ON sc.TargetOrgUnitID = sec.UniqueID
            WHERE admin.Name LIKE 'TEST%'
            GROUP BY admin.Name, dept.Name, sec.Name
            ORDER BY admin.Name, dept.Name, sec.Name
        """)
        
        print(f"\n[TEST HIERARCHY WITH SUBCASES]")
        for row in cursor.fetchall():
            print(f"  {row.Admin} -> {row.Dept} -> {row.Section}: {row.Subcases or 0} subcases")
        
        print(f"\n✅ Test data verification complete!")
        
    finally:
        cursor.close()
        conn.close()


def cleanup_test_data():
    """Clean up test data"""
    print("\n" + "="*80)
    print("CLEANUP OPTIONS")
    print("="*80)
    
    response = input("\nDo you want to clean up the test data? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("Test data preserved for testing.")
        return
    
    conn, cursor = get_db_cursor()
    try:
        # Delete action items first (FK constraint)
        cursor.execute("""
            DELETE ai
            FROM dbo.APP_SubcaseActionItem ai
            INNER JOIN dbo.APP_AdministrativeSubcase sc ON ai.SubcaseID = sc.SubcaseID
            INNER JOIN dbo.AdminsrationUnit u ON sc.TargetOrgUnitID = u.UniqueID
            WHERE u.Name LIKE 'TEST%'
        """)
        action_count = cursor.rowcount
        
        # Delete subcases
        cursor.execute("""
            DELETE sc
            FROM dbo.APP_AdministrativeSubcase sc
            INNER JOIN dbo.AdminsrationUnit u ON sc.TargetOrgUnitID = u.UniqueID
            WHERE u.Name LIKE 'TEST%'
        """)
        subcase_count = cursor.rowcount
        
        # Delete org units (cascading from children to parents)
        cursor.execute("DELETE FROM dbo.AdminsrationUnit WHERE Name LIKE 'TEST%'")
        orgunit_count = cursor.rowcount
        
        conn.commit()
        
        print(f"\n✅ Cleanup complete:")
        print(f"  Deleted {action_count} action items")
        print(f"  Deleted {subcase_count} subcases")
        print(f"  Deleted {orgunit_count} org units")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Error during cleanup: {str(e)}")
    finally:
        cursor.close()
        conn.close()


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("STEP 3.15 — TEST DATA SETUP")
    print("="*80)
    
    try:
        # Check existing data
        existing = check_existing_data()
        
        # Create hierarchy if needed
        if not existing['has_hierarchy']:
            hierarchy_info = create_test_hierarchy()
        else:
            hierarchy_info = existing
        
        # Create test subcases
        subcase_ids = create_test_subcases(hierarchy_info)
        
        # Create test action items
        create_test_action_items(subcase_ids)
        
        # Verify
        verify_test_data()
        
        print("\n" + "="*80)
        print("SETUP COMPLETE!")
        print("="*80)
        print("\n✅ Test data is ready!")
        print("\nYou can now run: python test_step3_15_insight_service.py")
        print("\nThe test will use:")
        if 'admin_id' in hierarchy_info:
            print(f"  - Administration ID: {hierarchy_info['admin_id']}")
        if 'dept_id' in hierarchy_info:
            print(f"  - Department ID: {hierarchy_info['dept_id']}")
        if 'section_ids' in hierarchy_info:
            print(f"  - Section IDs: {hierarchy_info['section_ids']}")
        
        # Offer cleanup option
        cleanup_test_data()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
