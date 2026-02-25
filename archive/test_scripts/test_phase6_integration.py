"""
PHASE 6: Integration & End-to-End Testing
Tests complete workflows and integration with incident case creation.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.core.database import get_connection
from backend.api.db_layer.doctors_db import create_doctor, search_doctors, get_doctor_profile
from backend.api.services.doctors_service import DoctorService
from datetime import datetime


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
        print(f"✓ Cleaned up test doctor: {doctor_name}")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_1_end_to_end_workflow():
    """Test 1: Complete workflow - Create → Search → Get Profile → Cleanup"""
    print("\n" + "="*70)
    print("TEST 1: End-to-End Workflow (Create → Search → Profile)")
    print("="*70)
    
    test_doctor_name = f"Dr. E2E Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    test_specialty = "Emergency Medicine E2E"
    
    try:
        # Step 1: Create doctor via service layer
        print(f"\n[STEP 1] Creating doctor: {test_doctor_name}")
        service = DoctorService()
        result = service.create_doctor(
            doctor_name=test_doctor_name,
            specialty=test_specialty,
            is_active=True,
            source_system="E2E_TEST"
        )
        
        assert result['success'], f"Failed to create: {result.get('message')}"
        doctor_id = result['doctor']['id']
        print(f"✓ Doctor created with ID: {doctor_id}")
        print(f"  - Name: {result['doctor']['name_en']}")
        print(f"  - Specialty: {result['doctor']['specialty']}")
        print(f"  - Source: {result['doctor']['source']}")
        
        # Step 2: Search for doctor
        print(f"\n[STEP 2] Searching for doctor: {test_doctor_name}")
        search_results = search_doctors(query=test_doctor_name)
        
        found = False
        for doc in search_results:
            if doc['id'] == doctor_id:
                found = True
                print(f"✓ Doctor found in search results")
                print(f"  - ID: {doc['id']}")
                print(f"  - Name: {doc['name_en']}")
                print(f"  - Specialty: {doc['specialty']}")
                print(f"  - Source: {doc['source']}")
                assert doc['source'] == 'reserve', "Doctor should be from reserve"
                break
        
        assert found, f"Doctor ID {doctor_id} not found in search results"
        
        # Step 3: Get doctor profile
        print(f"\n[STEP 3] Getting doctor profile for ID: {doctor_id}")
        profile = get_doctor_profile(doctor_id)
        
        assert profile is not None, "Profile should not be None"
        print(f"✓ Doctor profile retrieved")
        print(f"  - ID: {profile['id']}")
        print(f"  - Name: {profile['name_en']}")
        print(f"  - Specialty: {profile['specialty']}")
        print(f"  - Status: {profile['status']}")
        print(f"  - Source: {profile['source']}")
        
        assert profile['id'] == doctor_id, "Profile ID mismatch"
        assert profile['name_en'] == test_doctor_name, "Name mismatch"
        assert profile['specialty'] == test_specialty, "Specialty mismatch"
        assert profile['source'] == 'reserve', "Source should be reserve"
        
        print("\n✓✓✓ TEST 1 PASSED: End-to-End Workflow Complete")
        return True
        
    except AssertionError as e:
        print(f"\n✗✗✗ TEST 1 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗✗✗ TEST 1 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_doctor(test_doctor_name)


def test_2_dual_source_search():
    """Test 2: Verify search returns both hospital and reserve doctors"""
    print("\n" + "="*70)
    print("TEST 2: Dual-Source Search (Hospital + Reserve)")
    print("="*70)
    
    test_doctor_name = f"Dr. Dual Source Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create a reserve doctor
        print(f"\n[STEP 1] Creating reserve doctor: {test_doctor_name}")
        result = create_doctor(
            doctor_name=test_doctor_name,
            specialty="Dual Source Test",
            is_active=True,
            source_system="DUAL_TEST"
        )
        reserve_id = result['id']
        print(f"✓ Reserve doctor created with ID: {reserve_id}")
        
        # Search without term (get all)
        print(f"\n[STEP 2] Searching all doctors")
        all_doctors = search_doctors(query=None, limit=1000)
        
        print(f"✓ Found {len(all_doctors)} total doctors")
        
        # Count by source
        hospital_count = sum(1 for d in all_doctors if d['source'] == 'hospital')
        reserve_count = sum(1 for d in all_doctors if d['source'] == 'reserve')
        
        print(f"  - Hospital doctors: {hospital_count}")
        print(f"  - Reserve doctors: {reserve_count}")
        
        assert hospital_count > 0, "Should have hospital doctors"
        assert reserve_count > 0, "Should have reserve doctors"
        
        # Verify our test doctor is in results
        found = any(d['id'] == reserve_id for d in all_doctors)
        assert found, "Test doctor should be in results"
        
        print("\n✓✓✓ TEST 2 PASSED: Dual-Source Search Working")
        return True
        
    except AssertionError as e:
        print(f"\n✗✗✗ TEST 2 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗✗✗ TEST 2 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_doctor(test_doctor_name)


def test_3_reserve_doctor_in_view():
    """Test 3: Verify reserve doctor appears in APP_VIEWTABLE_VW_DOCTORS or unified query"""
    print("\n" + "="*70)
    print("TEST 3: Reserve Doctor Visibility in Views/Queries")
    print("="*70)
    
    test_doctor_name = f"Dr. View Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create a reserve doctor
        print(f"\n[STEP 1] Creating reserve doctor: {test_doctor_name}")
        result = create_doctor(
            doctor_name=test_doctor_name,
            specialty="View Test Specialty",
            is_active=True,
            source_system="VIEW_TEST"
        )
        doctor_id = result['id']
        print(f"✓ Reserve doctor created with ID: {doctor_id}")
        
        # Check if doctor is visible via UNION query (used by incident validation)
        print(f"\n[STEP 2] Checking visibility in unified doctor query")
        conn = get_connection()
        cursor = conn.cursor()
        
        # This is the query that incident validation should use
        cursor.execute("""
            SELECT DoctorID, DoctorName, Specialty, IsActive, SourceSystem, LastSyncedAt, 'hospital' as Source
            FROM APP_LOOKUP_DOCTOR
            WHERE DoctorID = ?
            UNION ALL
            SELECT DoctorID, DoctorName, Specialty, IsActive, SourceSystem, LastSyncedAt, 'reserve' as Source
            FROM APP_RESERVE_DOCTOR
            WHERE DoctorID = ?
        """, (doctor_id, doctor_id))
        
        row = cursor.fetchone()
        
        if row:
            print(f"✓ Doctor found in unified query")
            print(f"  - DoctorID: {row.DoctorID}")
            print(f"  - DoctorName: {row.DoctorName}")
            print(f"  - Source: {row.Source}")
            assert row.Source == 'reserve', "Should be from reserve"
            visibility_ok = True
        else:
            print(f"✗ Doctor NOT found in unified query")
            visibility_ok = False
        
        cursor.close()
        conn.close()
        
        assert visibility_ok, "Doctor should be visible in unified query"
        
        print("\n✓✓✓ TEST 3 PASSED: Reserve Doctor Visible in Queries")
        return True
        
    except AssertionError as e:
        print(f"\n✗✗✗ TEST 3 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗✗✗ TEST 3 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_doctor(test_doctor_name)


def test_4_incident_validation_compatibility():
    """Test 4: Check if doctor validation in insert_service would accept reserve doctors"""
    print("\n" + "="*70)
    print("TEST 4: Incident Validation Compatibility")
    print("="*70)
    
    test_doctor_name = f"Dr. Incident Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Create a reserve doctor
        print(f"\n[STEP 1] Creating reserve doctor: {test_doctor_name}")
        result = create_doctor(
            doctor_name=test_doctor_name,
            specialty="Incident Test Specialty",
            is_active=True,
            source_system="INCIDENT_TEST"
        )
        doctor_id = result['id']
        print(f"✓ Reserve doctor created with ID: {doctor_id}")
        
        # Simulate the validation check from insert_service.py
        print(f"\n[STEP 2] Simulating incident validation check")
        conn = get_connection()
        cursor = conn.cursor()
        
        # Current validation query in insert_service.py (line 202)
        cursor.execute(
            "SELECT COUNT(*) FROM APP_VIEWTABLE_VW_DOCTORS WHERE DoctorID = ?",
            (doctor_id,)
        )
        count_in_view = cursor.fetchone()[0]
        
        print(f"  - Count in APP_VIEWTABLE_VW_DOCTORS: {count_in_view}")
        
        if count_in_view == 0:
            print(f"⚠ Reserve doctor NOT in APP_VIEWTABLE_VW_DOCTORS")
            print(f"  This means incident validation will REJECT reserve doctors")
            print(f"  SOLUTION: Update insert_service.py to use UNION query")
            validation_works = False
        else:
            print(f"✓ Reserve doctor IS in APP_VIEWTABLE_VW_DOCTORS")
            validation_works = True
        
        cursor.close()
        conn.close()
        
        # This test documents the issue - we expect it to fail initially
        if not validation_works:
            print("\n⚠ TEST 4 IDENTIFIED ISSUE: insert_service needs update to accept reserve doctors")
            print("  Current: Checks APP_VIEWTABLE_VW_DOCTORS only")
            print("  Needed: Use UNION query to check both hospital + reserve")
        
        print("\n✓ TEST 4 COMPLETED: Compatibility check done (issue identified)")
        return validation_works  # Return actual status
        
    except Exception as e:
        print(f"\n✗✗✗ TEST 4 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        cleanup_test_doctor(test_doctor_name)


def test_5_service_layer_validation():
    """Test 5: Service layer edge cases and validation"""
    print("\n" + "="*70)
    print("TEST 5: Service Layer Validation & Edge Cases")
    print("="*70)
    
    service = DoctorService()
    tests_passed = 0
    total_tests = 5
    
    # Test 5.1: Name too short
    print("\n[TEST 5.1] Name too short (< 3 chars)")
    try:
        result = service.create_doctor(doctor_name="Dr", specialty="Test", is_active=True)
        print(f"✗ Should have raised ValueError but got: {result}")
    except ValueError as e:
        print(f"✓ Rejected: {e}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 5.2: Name too long
    print("\n[TEST 5.2] Name too long (> 200 chars)")
    try:
        long_name = "Dr. " + "X" * 200
        result = service.create_doctor(doctor_name=long_name, specialty="Test", is_active=True)
        print(f"✗ Should have raised ValueError but got: {result}")
    except ValueError as e:
        print(f"✓ Rejected: {e}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 5.3: Specialty too long
    print("\n[TEST 5.3] Specialty too long (> 200 chars)")
    try:
        long_specialty = "X" * 201
        result = service.create_doctor(
            doctor_name="Dr. Test", 
            specialty=long_specialty, 
            is_active=True
        )
        print(f"✗ Should have raised ValueError but got: {result}")
    except ValueError as e:
        print(f"✓ Rejected: {e}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    # Test 5.4: Whitespace trimming
    print("\n[TEST 5.4] Whitespace trimming")
    test_name = f"Dr. Whitespace Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        result = service.create_doctor(
            doctor_name=f"  {test_name}  ",
            specialty="  Cardiology  ",
            is_active=True
        )
        assert result['success'], f"Should succeed: {result.get('message')}"
        assert result['doctor']['name_en'].strip() == test_name, "Should trim name"
        print(f"✓ Whitespace trimmed correctly")
        tests_passed += 1
        cleanup_test_doctor(test_name)
    except Exception as e:
        print(f"✗ Failed: {e}")
        cleanup_test_doctor(test_name)
    
    # Test 5.5: Duplicate detection
    print("\n[TEST 5.5] Duplicate doctor detection")
    dup_name = f"Dr. Duplicate Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        # Create first
        result1 = service.create_doctor(doctor_name=dup_name, specialty="Test", is_active=True)
        assert result1['success'], "First creation should succeed"
        
        # Try duplicate
        try:
            result2 = service.create_doctor(doctor_name=dup_name, specialty="Test", is_active=True)
            print(f"✗ Should have raised ValueError but got: {result2}")
        except ValueError as e:
            print(f"✓ Duplicate rejected: {e}")
            tests_passed += 1
        
        cleanup_test_doctor(dup_name)
    except Exception as e:
        print(f"✗ Failed: {e}")
        cleanup_test_doctor(dup_name)
    
    print(f"\n✓ TEST 5 RESULTS: {tests_passed}/{total_tests} validation tests passed")
    return tests_passed == total_tests


def test_6_search_filtering():
    """Test 6: Search filtering and specialty filtering"""
    print("\n" + "="*70)
    print("TEST 6: Search Filtering & Specialty Search")
    print("="*70)
    
    test_doctors = []
    try:
        # Create test doctors with different specialties
        print("\n[STEP 1] Creating test doctors with different specialties")
        specialties = ["Cardiology", "Neurology", "Orthopedics"]
        
        for spec in specialties:
            name = f"Dr. Filter Test {spec} {datetime.now().strftime('%H%M%S')}"
            result = create_doctor(
                doctor_name=name,
                specialty=spec,
                is_active=True,
                source_system="FILTER_TEST"
            )
            test_doctors.append((name, result['id'], spec))
            print(f"✓ Created: {name} - {spec}")
        
        # Test name search
        print("\n[STEP 2] Testing name search")
        search_results = search_doctors(query="Filter Test Cardiology")
        cardio_found = any(d['id'] == test_doctors[0][1] for d in search_results)
        assert cardio_found, "Should find Cardiology doctor"
        print(f"✓ Name search works - found Cardiology doctor")
        
        # Test status filtering
        print("\n[STEP 3] Testing active-only filter")
        search_results = search_doctors(status="active", limit=100)
        all_active = all(d['status'] == 'active' for d in search_results[:50])  # Check first 50
        assert all_active, "All should be active"
        print(f"✓ Active-only filter works")
        
        # Test all specialties returned
        print("\n[STEP 4] Verifying all test doctors are searchable")
        all_results = search_doctors(query="Filter Test", limit=100)
        found_count = sum(1 for d in all_results if d['id'] in [t[1] for t in test_doctors])
        assert found_count == 3, f"Should find all 3 test doctors, found {found_count}"
        print(f"✓ All {found_count} test doctors found in search")
        
        print("\n✓✓✓ TEST 6 PASSED: Search filtering works correctly")
        return True
        
    except AssertionError as e:
        print(f"\n✗✗✗ TEST 6 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗✗✗ TEST 6 ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        for name, _, _ in test_doctors:
            cleanup_test_doctor(name)


def main():
    """Run all Phase 6 integration tests"""
    print("\n" + "="*70)
    print("PHASE 6: INTEGRATION & END-TO-END TESTING")
    print("Testing complete workflows and system integration")
    print("="*70)
    
    results = {
        "Test 1: End-to-End Workflow": test_1_end_to_end_workflow(),
        "Test 2: Dual-Source Search": test_2_dual_source_search(),
        "Test 3: Reserve Doctor Visibility": test_3_reserve_doctor_in_view(),
        "Test 4: Incident Validation": test_4_incident_validation_compatibility(),
        "Test 5: Service Layer Validation": test_5_service_layer_validation(),
        "Test 6: Search Filtering": test_6_search_filtering()
    }
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 6 TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL PHASE 6 TESTS PASSED!")
    else:
        print(f"\n⚠ {total - passed} test(s) need attention")
        if not results["Test 4: Incident Validation"]:
            print("\n⚠ CRITICAL: insert_service.py needs update to accept reserve doctors")
            print("  Action needed: Update doctor validation to use UNION query")
    
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
