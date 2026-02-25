"""
Test incident creation with reserve doctors
Validates that reserve doctors can be used when creating incident cases.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.core.database import get_connection
from backend.api.db_layer.doctors_db import create_doctor, search_doctors
from backend.api.services.insert_service import create_record
from datetime import datetime, date


def cleanup_test_doctor(doctor_name):
    """Remove test doctor from reserve table."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM APP_RESERVE_DOCTOR WHERE DoctorName = ?",
            (doctor_name,)
        )
        conn.commit()
        print(f"  ✓ Cleaned up doctor: {doctor_name}")
    except Exception as e:
        print(f"  ✗ Cleanup failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def cleanup_test_case(case_id):
    """Remove test incident case."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Delete from child tables first
        cursor.execute("DELETE FROM APP_IncidentCaseDoctor WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("DELETE FROM APP_IncidentCase_TargetDepartment WHERE IncidentRequestCaseID = ?", (case_id,))
        cursor.execute("DELETE FROM dbo.APP_IncidentCase WHERE IncidentRequestCaseID = ?", (case_id,))
        
        conn.commit()
        print(f"  ✓ Cleaned up case ID: {case_id}")
    except Exception as e:
        print(f"  ✗ Case cleanup failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_incident_with_reserve_doctor():
    """Test creating an incident case with a reserve doctor"""
    print("\n" + "="*70)
    print("TEST: Incident Creation with Reserve Doctor")
    print("="*70)
    
    test_doctor_name = f"Dr. Incident Integration {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    case_id = None
    
    try:
        # Step 1: Create reserve doctor
        print(f"\n[STEP 1] Creating reserve doctor: {test_doctor_name}")
        doctor_result = create_doctor(
            doctor_name=test_doctor_name,
            specialty="Emergency Medicine",
            is_active=True,
            source_system="INCIDENT_INTEGRATION_TEST"
        )
        doctor_id = doctor_result['id']
        print(f"✓ Reserve doctor created with ID: {doctor_id}")
        print(f"  - Name: {doctor_result['name_en']}")
        print(f"  - Specialty: {doctor_result['specialty']}")
        print(f"  - Source: {doctor_result['source']}")
        
        # Step 2: Verify doctor is searchable
        print(f"\n[STEP 2] Verifying doctor is searchable")
        search_results = search_doctors(query=test_doctor_name)
        found = any(d['id'] == doctor_id for d in search_results)
        assert found, "Doctor should be searchable"
        print(f"✓ Doctor found in search results")
        
        # Step 3: Create incident case with reserve doctor
        print(f"\n[STEP 3] Creating incident case with reserve doctor")
        
        # Get ACTUAL valid IDs from database
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get first valid ID from each table
        cursor.execute("SELECT TOP 1 DomainID FROM dbo.APP_LOOKUP_DOMAIN ORDER BY DomainID")
        domain_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY WHERE DomainID = ? ORDER BY CategoryID", (domain_id,))
        category_row = cursor.fetchone()
        category_id = category_row[0] if category_row else None
        if not category_id:
            # Get any category
            cursor.execute("SELECT TOP 1 CategoryID FROM dbo.APP_LOOKUP_CATEGORY ORDER BY CategoryID")
            category_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY WHERE CategoryID = ? ORDER BY SubCategoryID", (category_id,))
        subcategory_row = cursor.fetchone()
        subcategory_id = subcategory_row[0] if subcategory_row else None
        if not subcategory_id:
            cursor.execute("SELECT TOP 1 SubCategoryID FROM dbo.APP_LOOKUP_SUBCATEGORY ORDER BY SubCategoryID")
            subcategory_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION WHERE SubCategoryID = ? ORDER BY ClassificationID", (subcategory_id,))
        classification_row = cursor.fetchone()
        classification_id = classification_row[0] if classification_row else None
        if not classification_id:
            cursor.execute("SELECT TOP 1 ClassificationID FROM dbo.APP_LOOKUP_CLASSIFICATION ORDER BY ClassificationID")
            classification_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 SeverityID FROM dbo.APP_LOOKUP_SEVERITY ORDER BY SeverityID")
        severity_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 StageID FROM dbo.APP_LOOKUP_CASE_STAGE ORDER BY StageID")
        stage_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 HarmID FROM dbo.APP_LOOKUP_HARM_LEVEL ORDER BY HarmID")
        harm_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 BuildingID FROM dbo.APP_LOOKUP_BUILDING ORDER BY BuildingID")
        building_id = cursor.fetchone()[0]
        
        cursor.execute("SELECT TOP 1 SourceID FROM dbo.APP_LOOKUP_SOURCE ORDER BY SourceID")
        source_id = cursor.fetchone()[0]
        
        # Use hardcoded department that exists in other tests
        issuing_department_id = 43
        
        cursor.close()
        conn.close()
        
        # Create incident data with reserve doctor
        incident_data = {
            'complaint_text': f'Test incident with reserve doctor {test_doctor_name}',
            'feedback_received_date': date.today().isoformat(),
            'issuing_department_id': issuing_department_id,
            'domain_id': domain_id,
            'category_id': category_id,
            'subcategory_id': subcategory_id,
            'classification_id': classification_id,
            'severity_id': severity_id,
            'stage_id': stage_id,
            'harm_id': harm_id,
            'requires_explanation': False,
            'clinical_risk_type_id': 1,  # Ordinary
            'feedback_intent_type_id': 1,
            'immediate_action': 'Test action',
            'taken_action': 'Test action taken',
            'patient_name': 'Test Patient',
            'is_inpatient': True,
            'source_id': source_id,
            'building_id': building_id,
            'doctors': [
                {
                    'doctor_id': doctor_id,
                    'doctor_name': test_doctor_name
                }
            ],
            'target_departments': [issuing_department_id]
        }
        
        result = create_record(incident_data)
        
        if result.get('success'):
            case_id = result.get('id') or result.get('incident_id')  # Try both field names
            print(f"✓ Incident case created successfully!")
            print(f"  - Case ID: {case_id}")
            print(f"  - Message: {result.get('message')}")
            
            # Step 4: Verify doctor was linked to case
            print(f"\n[STEP 4] Verifying doctor linkage to case")
            conn = get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DoctorID, DoctorName
                FROM APP_IncidentCaseDoctor
                WHERE IncidentRequestCaseID = ? AND DoctorID = ?
            """, (case_id, doctor_id))
            
            case_doctor = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if case_doctor:
                print(f"✓ Reserve doctor successfully linked to incident case")
                print(f"  - DoctorID: {case_doctor.DoctorID}")
                print(f"  - DoctorName: {case_doctor.DoctorName}")
                test_passed = True
            else:
                print(f"✗ Doctor not found in case linkage")
                test_passed = False
        else:
            print(f"✗ Incident case creation failed:")
            print(f"  - Error: {result.get('error')}")
            print(f"  - Message: {result.get('message')}")
            print(f"  - Field: {result.get('field')}")
            test_passed = False
        
        if test_passed:
            print("\n✓✓✓ TEST PASSED: Reserve doctors work in incident creation!")
        else:
            print("\n✗✗✗ TEST FAILED: Reserve doctor integration issue")
        
        return test_passed
        
    except AssertionError as e:
        print(f"\n✗✗✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗✗✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print(f"\n[CLEANUP]")
        if case_id:
            cleanup_test_case(case_id)
        cleanup_test_doctor(test_doctor_name)


def test_incident_validation_rejects_invalid_doctor():
    """Test that invalid doctor IDs are properly rejected"""
    print("\n" + "="*70)
    print("TEST: Incident Validation Rejects Invalid Doctor")
    print("="*70)
    
    try:
        print(f"\n[STEP 1] Attempting to create incident with non-existent doctor ID 999999")
        
        # Use simple, known-good reference IDs (same as first test)
        incident_data = {
            'complaint_text': 'Test incident with invalid doctor',
            'feedback_received_date': date.today().isoformat(),
            'issuing_department_id': 43,
            'domain_id': 1,
            'category_id': 1,
            'subcategory_id': 1,
            'classification_id': 1,
            'severity_id': 1,
            'stage_id': 1,
            'harm_id': 1,
            'requires_explanation': False,
            'clinical_risk_type_id': 1,
            'feedback_intent_type_id': 1,
            'immediate_action': 'Test action',
            'taken_action': 'Test action taken',
            'patient_name': 'Test Patient',
            'is_inpatient': True,
            'source_id': 4,
            'building_id': 1,
            'doctors': [
                {
                    'doctor_id': 999999,
                    'doctor_name': 'Non-existent Doctor'
                }
            ],
            'target_departments': [43]
        }
        
        result = create_record(incident_data)
        
        if not result.get('success') and result.get('error') == 'INVALID_REFERENCE':
            print(f"✓ Invalid doctor ID properly rejected")
            print(f"  - Error: {result.get('error')}")
            print(f"  - Message: {result.get('message')}")
            print(f"  - Field: {result.get('field')}")
            print("\n✓✓✓ TEST PASSED: Validation working correctly")
            return True
        else:
            print(f"✗ Expected rejection but got: {result}")
            print("\n✗✗✗ TEST FAILED: Invalid doctor was not rejected")
            return False
        
    except Exception as e:
        print(f"\n✗✗✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all incident integration tests"""
    print("\n" + "="*70)
    print("INCIDENT INTEGRATION TESTS")
    print("Testing reserve doctors in incident case creation")
    print("="*70)
    
    results = {
        "Test 1: Incident with Reserve Doctor": test_incident_with_reserve_doctor(),
        "Test 2: Validation Rejects Invalid Doctor": test_incident_validation_rejects_invalid_doctor()
    }
    
    # Summary
    print("\n" + "="*70)
    print("INCIDENT INTEGRATION TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL INCIDENT INTEGRATION TESTS PASSED!")
        print("\n✓ Reserve doctors can be used in incident case creation")
        print("✓ Validation properly checks both hospital and reserve tables")
        print("✓ Invalid doctor IDs are properly rejected")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
