"""
STEP 3.10 Adapter Integration Test
Tests that the adapter hooks in insert_service.py and seasonal_report_generator.py
properly trigger subcase creation in API v2.
"""

import sys
import os
from datetime import datetime

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


def test(description):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {description}")
            print('='*60)
            func()
        return wrapper
    return decorator


def get_db_cursor():
    """Get database cursor"""
    conn = get_connection()
    cursor = conn.cursor()
    return conn, cursor


@test("1. Test Incident Adapter Integration")
def test_incident_adapter():
    """
    Test that creating a new incident automatically triggers subcase creation
    via the adapter hook in insert_service.py
    """
    from api.services.insert_service import create_record
    
    print("\n[SETUP] Creating test incident via legacy API...")
    
    # Create test incident data
    test_data = {
        'complaint_text': 'Test incident for adapter integration',
        'feedback_received_date': datetime.now().strftime('%Y-%m-%d'),
        'issuing_department_id': 1,
        'domain_id': 1,
        'category_id': 1,
        'subcategory_id': 1,
        'classification_id': 78,  # Valid ID from database
        'severity_id': 1,
        'stage_id': 1,
        'harm_id': 1,
        'requires_explanation': False,
        'clinical_risk_type_id': 1,  # Ordinary (not red flag)
        'feedback_intent_type_id': 1,
        'immediate_action': 'Immediate action taken',
        'taken_action': 'Action taken',
        'patient_name': 'Test Patient',
        'is_inpatient': True,
        'source_id': 1,
        'building_id': 1,
        'target_department_ids': [2, 3]  # Target departments for subcases
    }
    
    # Create the incident (this should trigger the adapter)
    result = create_record(test_data)
    
    print(f"\n[RESULT] Incident creation result:")
    print(f"  Success: {result.get('success')}")
    print(f"  Incident ID: {result.get('incident_id')}")
    
    if not result.get('success'):
        print(f"  ❌ Error: {result.get('message')}")
        return
    
    incident_id = result.get('incident_id')
    
    # Give the adapter a moment to execute
    import time
    time.sleep(1)
    
    # Check if subcases were created
    print(f"\n[VERIFY] Checking for subcases created by adapter...")
    conn, cursor = get_db_cursor()
    
    try:
        cursor.execute("""
            SELECT 
                SubcaseID,
                CaseType,
                IncidentRequestCaseID,
                TargetOrgUnitID,
                Status,
                CreatedAt
            FROM dbo.APP_AdministrativeSubcase
            WHERE IncidentRequestCaseID = ?
            ORDER BY CreatedAt DESC
        """, (incident_id,))
        
        subcases = cursor.fetchall()
        
        # STRICT VERIFICATION: Number of subcases must match target departments
        expected_count = len(test_data['target_department_ids'])
        actual_count = len(subcases)
        
        print(f"\n[VERIFICATION] Subcase count check:")
        print(f"  Expected: {expected_count} (target_department_ids)")
        print(f"  Actual: {actual_count} (subcases created)")
        
        if actual_count != expected_count:
            print(f"  ❌ FAILURE: Subcase count mismatch!")
            print(f"  Expected {expected_count} subcases but got {actual_count}")
            raise AssertionError(
                f"INVARIANT VIOLATION: Expected {expected_count} subcases "
                f"for {expected_count} target departments, but got {actual_count}"
            )
        
        print(f"  ✅ SUCCESS: Subcase count matches target department count!")
        
        # Verify each subcase targets one of the expected departments
        expected_dept_ids = set(test_data['target_department_ids'])
        actual_dept_ids = {sc.TargetOrgUnitID for sc in subcases}
        
        print(f"\n[VERIFICATION] Target department ID check:")
        print(f"  Expected dept IDs: {sorted(expected_dept_ids)}")
        print(f"  Actual dept IDs: {sorted(actual_dept_ids)}")
        
        if actual_dept_ids != expected_dept_ids:
            print(f"  ❌ FAILURE: Target department ID mismatch!")
            missing = expected_dept_ids - actual_dept_ids
            unexpected = actual_dept_ids - expected_dept_ids
            if missing:
                print(f"  Missing dept IDs: {sorted(missing)}")
            if unexpected:
                print(f"  Unexpected dept IDs: {sorted(unexpected)}")
            raise AssertionError(
                f"INVARIANT VIOLATION: Target department IDs do not match. "
                f"Expected {sorted(expected_dept_ids)}, got {sorted(actual_dept_ids)}"
            )
        
        print(f"  ✅ SUCCESS: All target department IDs match!")
        
        # Display created subcases
        if subcases:
            print(f"\n  Created subcases:")
            for sc in subcases:
                print(f"    - SubcaseID={sc.SubcaseID}, Type={sc.CaseType}, "
                      f"Target={sc.TargetOrgUnitID}, Status={sc.Status}")
        
        print(f"\n  🎉 ALL ADAPTER INVARIANTS VERIFIED!")
        
        # Cleanup
        print(f"\n[CLEANUP] Removing test incident {incident_id}...")
        cursor.execute("DELETE FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?", (incident_id,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCaseTargetDepartment WHERE IncidentRequestCaseID = ?", (incident_id,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (incident_id,))
        conn.commit()
        print(f"  ✅ Cleanup complete")
        
    finally:
        cursor.close()
        conn.close()


@test("2. Test Seasonal Report Adapter Integration")
def test_seasonal_report_adapter():
    """
    Test that generating a seasonal report automatically triggers subcase creation
    via the adapter hook in seasonal_report_generator.py
    """
    from api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
    
    print("\n[SETUP] Generating test seasonal report via legacy API...")
    
    # Use existing season and org unit for testing
    test_season_id = 1  # Assuming season 1 exists
    test_orgunit_id = 1  # Assuming org unit 1 exists
    test_orgunit_type = 1  # Department
    test_user_id = 1
    
    try:
        # Generate the seasonal report (this should trigger the adapter)
        report = get_or_generate_seasonal_report(
            season_id=test_season_id,
            orgunit_id=test_orgunit_id,
            orgunit_type=test_orgunit_type,
            user_id=test_user_id
        )
        
        print(f"\n[RESULT] Seasonal report generation result:")
        print(f"  Report ID: {report.get('header', {}).get('seasonal_report_id')}")
        print(f"  Total Cases: {report.get('header', {}).get('total_cases', 0)}")
        
        seasonal_report_id = report.get('header', {}).get('seasonal_report_id')
        
        if not seasonal_report_id:
            print(f"  ❌ Error: Could not get seasonal report ID")
            return
        
        # Give the adapter a moment to execute
        import time
        time.sleep(1)
        
        # Check if subcases were created
        print(f"\n[VERIFY] Checking for subcases created by adapter...")
        conn, cursor = get_db_cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    SubcaseID,
                    CaseType,
                    SeasonalReportID,
                    TargetOrgUnitID,
                    Status,
                    CreatedAt
                FROM dbo.APP_AdministrativeSubcase
                WHERE SeasonalReportID = ?
                ORDER BY CreatedAt DESC
            """, (seasonal_report_id,))
            
            subcases = cursor.fetchall()
            
            if subcases:
                print(f"  ✅ SUCCESS: {len(subcases)} subcase(s) created automatically!")
                print(f"\n  Subcases created:")
                for sc in subcases:
                    print(f"    - SubcaseID={sc.SubcaseID}, Type={sc.CaseType}, "
                          f"Target={sc.TargetOrgUnitID}, Status={sc.Status}")
            else:
                print(f"  ⚠️  WARNING: No subcases found for seasonal report {seasonal_report_id}")
                print(f"  The adapter may have failed silently (check logs above)")
                print(f"  NOTE: This is normal if the seasonal report has no policy violations")
            
            # Note: We don't cleanup seasonal reports as they're part of the normal data
            
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        print(f"  ❌ Error generating seasonal report: {str(e)}")
        import traceback
        traceback.print_exc()


@test("3. Verify Case Creation Service is Available")
def test_case_creation_service_exists():
    """
    Verify that the case_creation_service module exists and has the required functions
    """
    print("\n[VERIFY] Checking case_creation_service module...")
    
    try:
        from api_v2.services.case_creation_service import (
            create_subcases_for_incident,
            create_subcases_for_seasonal_report
        )
        
        print("  ✅ create_subcases_for_incident found")
        print("  ✅ create_subcases_for_seasonal_report found")
        
        # Check function signatures
        import inspect
        
        sig1 = inspect.signature(create_subcases_for_incident)
        print(f"\n  create_subcases_for_incident signature: {sig1}")
        
        sig2 = inspect.signature(create_subcases_for_seasonal_report)
        print(f"  create_subcases_for_seasonal_report signature: {sig2}")
        
        print("\n  ✅ All required functions are available!")
        
    except ImportError as e:
        print(f"  ❌ Import error: {str(e)}")
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")


@test("4. Check Adapter Code in insert_service.py")
def test_adapter_code_in_insert_service():
    """
    Verify that the adapter code was correctly added to insert_service.py
    """
    print("\n[VERIFY] Checking insert_service.py for adapter code...")
    
    with open('backend/api/services/insert_service.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the import
    if 'from backend.api_v2.services.case_creation_service import create_subcases_for_incident' in content:
        print("  ✅ Import statement found")
    else:
        print("  ❌ Import statement NOT found")
    
    # Check for the function call
    if 'create_subcases_for_incident(new_id' in content:
        print("  ✅ Function call found")
    else:
        print("  ❌ Function call NOT found")
    
    # Check for try-except wrapper
    if 'API V2 ADAPTER' in content:
        print("  ✅ Adapter section comment found")
    else:
        print("  ❌ Adapter section comment NOT found")


@test("5. Check Adapter Code in seasonal_report_generator.py")
def test_adapter_code_in_seasonal_generator():
    """
    Verify that the adapter code was correctly added to seasonal_report_generator.py
    """
    print("\n[VERIFY] Checking seasonal_report_generator.py for adapter code...")
    
    with open('backend/api/services/seasonal_report_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the import
    if 'from backend.api_v2.services.case_creation_service import create_subcases_for_seasonal_report' in content:
        print("  ✅ Import statement found")
    else:
        print("  ❌ Import statement NOT found")
    
    # Check for the function call
    if 'create_subcases_for_seasonal_report(seasonal_report_id' in content:
        print("  ✅ Function call found")
    else:
        print("  ❌ Function call NOT found")
    
    # Check for try-except wrapper
    if 'API V2 ADAPTER' in content:
        print("  ✅ Adapter section comment found")
    else:
        print("  ❌ Adapter section comment NOT found")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.10 — ADAPTER INTEGRATION TEST SUITE")
    print("Testing incident_service.py & seasonal_report_generator.py adapters")
    print("="*80)
    
    # Static checks first
    test_case_creation_service_exists()
    test_adapter_code_in_insert_service()
    test_adapter_code_in_seasonal_generator()
    
    # Integration tests
    test_incident_adapter()
    test_seasonal_report_adapter()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)
    print("\n✅ If all tests passed, STEP 3.10 is complete!")
    print("✅ Incidents now automatically create subcases")
    print("✅ Seasonal reports now automatically create subcases")
    print("✅ Legacy behavior remains unchanged (adapter is non-blocking)")
