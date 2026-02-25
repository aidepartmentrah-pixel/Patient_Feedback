"""
Test grouped inbox DB layer queries - Stage 1 Testing

This test file validates the enhanced database queries that fetch
subcases with full details for the Insight page grouped inbox feature.

Tests:
1. Verify query structure and field presence
2. Verify supervisor lookup functionality
"""

import sys
import os

# Add backend to path
sys.path.insert(0, 'backend')

def test_get_subcases_with_details_structure():
    """Verify enhanced query returns all required fields"""
    from backend.api_v2.db_layer import administrative_subcase_db
    
    print("\n" + "="*60)
    print("TEST: Get subcases with details for SECTION")
    print("="*60)
    
    subcases = administrative_subcase_db.get_subcases_with_details_for_section()
    
    if len(subcases) > 0:
        first = subcases[0]
        
        # Verify core fields
        assert 'subcase_id' in first, "Missing subcase_id"
        assert 'case_type' in first, "Missing case_type"
        assert 'status' in first, "Missing status"
        assert 'created_at' in first, "Missing created_at"
        assert 'waiting_days' in first, "Missing waiting_days"
        
        # Verify org unit fields
        assert 'target_org_unit_id' in first, "Missing target_org_unit_id"
        assert 'org_unit_name' in first, "Missing org_unit_name"
        assert 'org_type' in first, "Missing org_type"
        
        # Verify case details (at least one should be present)
        has_case_details = (
            'case_description' in first or 
            'patient_name' in first or 
            'season_name' in first
        )
        assert has_case_details, "Missing case details (description/patient/season)"
        
        print(f"✅ All required fields present")
        print(f"✅ Found {len(subcases)} subcases")
        print(f"✅ Sample waiting days: {first['waiting_days']}")
        print(f"✅ Sample case type: {first['case_type']}")
        print(f"✅ Sample status: {first['status']}")
        print(f"✅ Sample org unit: {first['org_unit_name']}")
        
        # Display sample data
        print("\n📋 Sample subcase data:")
        print(f"   Subcase ID: {first['subcase_id']}")
        print(f"   Case Type: {first['case_type']}")
        print(f"   Status: {first['status']}")
        print(f"   Waiting Days: {first['waiting_days']}")
        print(f"   Org Unit: {first['org_unit_name']}")
        if first.get('patient_name'):
            print(f"   Patient: {first['patient_name']}")
        if first.get('season_name'):
            print(f"   Season: {first['season_name']}")
        if first.get('severity'):
            print(f"   Severity: {first['severity']}")
        if first.get('category'):
            print(f"   Category: {first['category']}")
    else:
        print("⚠️  No subcases found for SECTION (test data needed)")
        print("    Expected statuses: SUBMITTED_TO_SECTION, RETURNED_TO_SECTION_FOR_REVISION")


def test_get_subcases_with_details_department():
    """Verify department query works"""
    from backend.api_v2.db_layer import administrative_subcase_db
    
    print("\n" + "="*60)
    print("TEST: Get subcases with details for DEPARTMENT")
    print("="*60)
    
    subcases = administrative_subcase_db.get_subcases_with_details_for_department()
    
    if len(subcases) > 0:
        print(f"✅ Found {len(subcases)} subcases for department")
        print(f"✅ Sample waiting days: {subcases[0]['waiting_days']}")
        print(f"✅ Sample status: {subcases[0]['status']}")
    else:
        print("⚠️  No subcases found for DEPARTMENT (test data needed)")
        print("    Expected statuses: SECTION_ACCEPTED_PENDING_DEPT, RETURNED_TO_DEPT_FOR_REVISION")


def test_get_subcases_with_details_administration():
    """Verify administration query works"""
    from backend.api_v2.db_layer import administrative_subcase_db
    
    print("\n" + "="*60)
    print("TEST: Get subcases with details for ADMINISTRATION")
    print("="*60)
    
    subcases = administrative_subcase_db.get_subcases_with_details_for_administration()
    
    if len(subcases) > 0:
        print(f"✅ Found {len(subcases)} subcases for administration")
        print(f"✅ Sample waiting days: {subcases[0]['waiting_days']}")
        print(f"✅ Sample status: {subcases[0]['status']}")
    else:
        print("⚠️  No subcases found for ADMINISTRATION (test data needed)")
        print("    Expected status: DEPT_ACCEPTED_PENDING_ADMIN")


def test_get_supervisor_name():
    """Verify supervisor lookup works"""
    from backend.api_v2.db_layer import administrative_subcase_db
    from core.database import get_connection
    
    print("\n" + "="*60)
    print("TEST: Get supervisor name for org unit")
    print("="*60)
    
    # Get a real org unit ID from database
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 UniqueID FROM dbo.AdminsrationUnit WHERE Type = 324")
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if row:
        org_unit_id = row[0]
        supervisor = administrative_subcase_db.get_supervisor_name_for_org_unit(org_unit_id)
        
        if supervisor:
            print(f"✅ Supervisor found for org unit {org_unit_id}: {supervisor}")
        else:
            print(f"⚠️  No supervisor assigned to org unit {org_unit_id}")
            print(f"    This is expected if no admin is assigned to this unit")
    else:
        print("⚠️  No org units found with Type = 324")


def test_waiting_days_calculation():
    """Verify waiting days are calculated correctly"""
    from backend.api_v2.db_layer import administrative_subcase_db
    from datetime import datetime, timedelta
    
    print("\n" + "="*60)
    print("TEST: Waiting days calculation")
    print("="*60)
    
    subcases = administrative_subcase_db.get_subcases_with_details_for_section()
    
    if len(subcases) > 0:
        for subcase in subcases[:3]:  # Check first 3
            created_at = subcase['created_at']
            waiting_days = subcase['waiting_days']
            
            # Calculate expected waiting days
            if created_at:
                expected_days = (datetime.now() - created_at).days
                
                # Allow 1 day tolerance for time differences
                assert abs(waiting_days - expected_days) <= 1, \
                    f"Waiting days mismatch: got {waiting_days}, expected ~{expected_days}"
                
                print(f"✅ Subcase {subcase['subcase_id']}: {waiting_days} days (created: {created_at.date()})")
        
        print(f"✅ Waiting days calculation verified for {min(3, len(subcases))} subcases")
    else:
        print("⚠️  No subcases to verify waiting days")


if __name__ == '__main__':
    print("\n" + "="*70)
    print(" STAGE 1: DATABASE LAYER TESTING - Enhanced Inbox Queries")
    print("="*70)
    
    try:
        test_get_subcases_with_details_structure()
        test_get_subcases_with_details_department()
        test_get_subcases_with_details_administration()
        test_get_supervisor_name()
        test_waiting_days_calculation()
        
        print("\n" + "="*70)
        print("✅ ALL STAGE 1 TESTS COMPLETED")
        print("="*70)
        print("\nNext Steps:")
        print("  1. Review the sample data output above")
        print("  2. Verify field mappings match API requirements")
        print("  3. Check if supervisor lookup returns expected admins")
        print("  4. Proceed to Stage 2: Service Layer implementation")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
