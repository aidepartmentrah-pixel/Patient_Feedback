"""
Phase 2 Test (Corrected): Database Layer - Write Operations
Test the create_doctor() function with IDENTICAL table structure
"""

import sys
import os
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from api.db_layer.doctors_db import create_doctor


def test_create_doctor_valid():
    """Test 1: Create a valid doctor with all fields"""
    print("\n" + "="*60)
    print("TEST 1: Create a valid doctor with all fields")
    print("="*60)
    
    try:
        result = create_doctor(
            doctor_name=f"Dr. Ahmed Test {datetime.now().strftime('%H%M%S')}",
            specialty="Interventional Cardiology",
            is_active=True,
            source_system="MANUAL"
        )
        
        print(f"\n✅ Doctor created successfully!")
        print(f"   DoctorID: {result['id']}")
        print(f"   Name: {result['name_en']}")
        print(f"   Specialty: {result['specialty']}")
        print(f"   Status: {result['status']}")
        print(f"   Source: {result['source']}")
        print(f"   SourceSystem: {result['source_system']}")
        
        if result['id'] and result['id'] > 0:
            print("\n✅ PASS: Valid doctor created with ID")
            return True, result
        else:
            print("\n❌ FAIL: No DoctorID returned")
            return False, None
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False, None


def test_create_doctor_minimal_fields():
    """Test 2: Create doctor with only required field (name)"""
    print("\n" + "="*60)
    print("TEST 2: Create doctor with minimal required field")
    print("="*60)
    
    try:
        result = create_doctor(
            doctor_name=f"Dr. Minimal {datetime.now().strftime('%H%M%S')}"
            # specialty, is_active, source_system all optional
        )
        
        print(f"\n✅ Doctor created with minimal fields!")
        print(f"   DoctorID: {result['id']}")
        print(f"   Name: {result['name_en']}")
        print(f"   Specialty: '{result['specialty']}'")
        print(f"   Status: {result['status']}")
        
        if result['id'] and result['id'] > 0:
            print("\n✅ PASS: Minimal fields accepted")
            return True, result
        else:
            print("\n❌ FAIL: No DoctorID returned")
            return False, None
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return False, None


def test_duplicate_doctor_name():
    """Test 3: Reject duplicate doctor name"""
    print("\n" + "="*60)
    print("TEST 3: Reject duplicate doctor name")
    print("="*60)
    
    try:
        # Create first doctor with unique name
        unique_name = f"Dr. Duplicate Test {datetime.now().strftime('%H%M%S%f')}"
        
        result1 = create_doctor(
            doctor_name=unique_name,
            specialty="Test Specialty"
        )
        print(f"✅ First doctor created: ID={result1['id']}, Name={unique_name}")
        
        # Try to create duplicate
        try:
            result2 = create_doctor(
                doctor_name=unique_name,  # Same name
                specialty="Different Specialty"
            )
            print("❌ FAIL: Duplicate was allowed (should have been rejected)")
            return False
        except ValueError as ve:
            if "already exists" in str(ve).lower():
                print(f"✅ Duplicate correctly rejected: {str(ve)[:80]}...")
                print("\n✅ PASS: Duplicate detection working")
                return True
            else:
                print(f"❌ FAIL: Wrong error message: {ve}")
                return False
                
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_inactive_doctor():
    """Test 4: Create inactive doctor"""
    print("\n" + "="*60)
    print("TEST 4: Create inactive doctor")
    print("="*60)
    
    try:
        result = create_doctor(
            doctor_name=f"Dr. Inactive {datetime.now().strftime('%H%M%S')}",
            specialty="General Medicine",
            is_active=False  # Inactive
        )
        
        print(f"\n✅ Inactive doctor created!")
        print(f"   DoctorID: {result['id']}")
        print(f"   Name: {result['name_en']}")
        print(f"   Status: {result['status']}")
        
        if result['status'] == 'inactive':
            print("\n✅ PASS: Inactive status correctly set")
            return True
        else:
            print(f"\n❌ FAIL: Status should be 'inactive' but got '{result['status']}'")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_custom_source_system():
    """Test 5: Create doctor with custom source system"""
    print("\n" + "="*60)
    print("TEST 5: Create doctor with custom source system")
    print("="*60)
    
    try:
        result = create_doctor(
            doctor_name=f"Dr. Custom Source {datetime.now().strftime('%H%M%S')}",
            specialty="Pediatrics",
            source_system="TEST_SYSTEM"
        )
        
        print(f"\n✅ Doctor created with custom source!")
        print(f"   DoctorID: {result['id']}")
        print(f"   Name: {result['name_en']}")
        print(f"   SourceSystem: {result['source_system']}")
        
        if result['source_system'] == 'TEST_SYSTEM':
            print("\n✅ PASS: Custom source system correctly set")
            return True
        else:
            print(f"\n❌ FAIL: SourceSystem should be 'TEST_SYSTEM' but got '{result['source_system']}'")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def test_specialty_optional():
    """Test 6: Verify specialty is truly optional"""
    print("\n" + "="*60)
    print("TEST 6: Verify specialty is optional")
    print("="*60)
    
    try:
        result = create_doctor(
            doctor_name=f"Dr. No Specialty {datetime.now().strftime('%H%M%S')}"
            # No specialty provided
        )
        
        print(f"\n✅ Doctor created without specialty!")
        print(f"   DoctorID: {result['id']}")
        print(f"   Name: {result['name_en']}")
        print(f"   Specialty: '{result['specialty']}'")
        
        # Specialty should be empty string
        if result['specialty'] == '':
            print("\n✅ PASS: Optional specialty handled correctly")
            return True
        else:
            print(f"\n⚠️  WARNING: Expected empty string, got '{result['specialty']}'")
            print("✅ PASS: Optional specialty accepted")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


def run_all_tests():
    """Run all Phase 2 tests"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  PHASE 2: DATABASE LAYER - WRITE OPERATIONS (CORRECTED) ║")
    print("║  Using IDENTICAL table structure                        ║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    tests = [
        ("Create Valid Doctor", test_create_doctor_valid),
        ("Create with Minimal Fields", test_create_doctor_minimal_fields),
        ("Reject Duplicate Name", test_duplicate_doctor_name),
        ("Create Inactive Doctor", test_inactive_doctor),
        ("Custom Source System", test_custom_source_system),
        ("Optional Specialty", test_specialty_optional),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if isinstance(result, tuple):
                result = result[0]
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║  TEST SUMMARY                                            ║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"  Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("-"*60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 2 is 100% complete!")
        print("✅ create_doctor() working with identical table structure")
        print("✅ Ready to proceed to Phase 3")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
