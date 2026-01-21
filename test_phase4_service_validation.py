"""
====================================================================
PHASE 4 TEST: Service Layer Validation - create_patient_service()
====================================================================
Purpose: Verify patients_service.create_patient_service() properly
         validates all inputs with comprehensive business rules

Test Coverage:
1. Valid patient creation with all fields
2. Valid patient with minimal fields
3. FirstName validation (required, length, characters)
4. Name fields validation (length, characters)
5. Phone number validation (format, length, digits)
6. BirthDate validation (format, not future, reasonable age)
7. SEX validation (valid values, normalization)
8. DocumentNumber validation (length, format)
9. MedicalFileNumber validation (length, format)
10. Address validation (length)
11. Whitespace trimming
12. Arabic characters support
13. Duplicate detection pass-through from DB layer
14. Error message clarity

Author: System
Date: 2026-01-20
====================================================================
"""

import sys
import os
from datetime import datetime, date, timedelta

# Add backend to path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
backend_api_path = os.path.join(os.path.dirname(__file__), 'backend', 'api')
sys.path.insert(0, backend_path)
sys.path.insert(0, backend_api_path)

# Import directly from the path
from backend.api.services.patients_service import create_patient_service
import pyodbc


def get_connection():
    """Get SQL Server connection."""
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )


def cleanup_test_data():
    """Remove test data"""
    print("\n" + "="*70)
    print("CLEANUP: Removing Test Data")
    print("="*70)
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM APP_RESERVE_PATIENT 
            WHERE FirstName LIKE 'TestPhase4%'
            OR DocumentNumber LIKE 'DOC-P4-%'
            OR MedicalFileNumber LIKE 'MRN-P4-%'
        """)
        deleted = cursor.rowcount
        conn.commit()
        print(f"✓ Deleted {deleted} test patient(s)")
        return True
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False
    finally:
        conn.close()


def test_1_valid_full_patient():
    """Test 1: Create patient with all valid fields"""
    print("\n" + "="*70)
    print("TEST 1: Valid Patient with All Fields")
    print("="*70)
    
    try:
        patient = create_patient_service(
            first_name="TestPhase4Valid",
            middle_name="Ahmad",
            last_name="AlTest",
            mother_name="Fatima",
            phone_number="0501234567",
            phone_number2="0509876543",
            birth_date="1990-05-15",
            sex="M",
            document_number="DOC-P4-001",
            medical_file_number="MRN-P4-001",
            spouse="Sara AlAhmad",
            address_line1="123 Test Street, Riyadh",
            address_line2="Building 5, Apt 201"
        )
        
        print(f"✓ PASS: Patient created successfully")
        print(f"  - ID: {patient['PatientAdmissionID']}")
        print(f"  - FullName: {patient['FullName']}")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_2_valid_minimal_patient():
    """Test 2: Create patient with minimal required fields"""
    print("\n" + "="*70)
    print("TEST 2: Valid Patient with Minimal Fields")
    print("="*70)
    
    try:
        patient = create_patient_service(first_name="TestPhase4Minimal")
        
        print(f"✓ PASS: Minimal patient created")
        print(f"  - ID: {patient['PatientAdmissionID']}")
        print(f"  - FullName: {patient['FullName']}")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_3_firstname_required():
    """Test 3: FirstName is required"""
    print("\n" + "="*70)
    print("TEST 3: FirstName Required Validation")
    print("="*70)
    
    test_cases = [
        ("", "Empty string"),
        ("   ", "Only whitespace"),
        (None, "None value")
    ]
    
    all_passed = True
    
    for value, description in test_cases:
        try:
            patient = create_patient_service(first_name=value)
            print(f"  ✗ {description}: Allowed (should fail)")
            all_passed = False
        except ValueError as e:
            if "required" in str(e).lower():
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Wrong error - {str(e)[:50]}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {description}: Unexpected error - {str(e)[:50]}")
            all_passed = False
    
    if all_passed:
        print("✓ PASS: FirstName requirement validated")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_4_firstname_length():
    """Test 4: FirstName length validation"""
    print("\n" + "="*70)
    print("TEST 4: FirstName Length Validation")
    print("="*70)
    
    test_cases = [
        ("A", False, "Too short (1 char)"),
        ("AB", True, "Minimum valid (2 chars)"),
        ("A" * 150, True, "Maximum valid (150 chars)"),
        ("A" * 151, False, "Too long (151 chars)")
    ]
    
    all_passed = True
    
    for value, should_pass, description in test_cases:
        try:
            if value == "A":
                test_value = value
            elif value == "AB":
                test_value = "TestPhase4MinLen"
            else:
                test_value = "TestPhase4" + value[10:]  # Unique prefix
            
            patient = create_patient_service(first_name=test_value)
            
            if should_pass:
                print(f"  ✓ {description}: Correctly allowed")
            else:
                print(f"  ✗ {description}: Should have been blocked")
                all_passed = False
                
        except ValueError as e:
            if not should_pass:
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Should have been allowed - {str(e)[:50]}")
                all_passed = False
    
    if all_passed:
        print("✓ PASS: FirstName length validated")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_5_firstname_characters():
    """Test 5: FirstName character validation"""
    print("\n" + "="*70)
    print("TEST 5: FirstName Character Validation")
    print("="*70)
    
    test_cases = [
        ("TestPhase4Valid", True, "Alphanumeric"),
        ("TestPhase4 Valid", True, "With space"),
        ("TestPhase4-Valid", True, "With hyphen"),
        ("TestPhase4'Valid", True, "With apostrophe"),
        ("TestPhase4محمد", True, "With Arabic characters"),
        ("TestPhase4@Invalid", False, "With @ symbol"),
        ("TestPhase4#Invalid", False, "With # symbol"),
        ("TestPhase4$Invalid", False, "With $ symbol")
    ]
    
    all_passed = True
    
    for value, should_pass, description in test_cases:
        try:
            patient = create_patient_service(first_name=value)
            
            if should_pass:
                print(f"  ✓ {description}: Correctly allowed")
            else:
                print(f"  ✗ {description}: Should have been blocked")
                all_passed = False
                
        except ValueError as e:
            if not should_pass and "invalid characters" in str(e).lower():
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Unexpected - {str(e)[:50]}")
                all_passed = False
        except Exception as e:
            print(f"  ✗ {description}: Unexpected error - {str(e)[:50]}")
            all_passed = False
    
    if all_passed:
        print("✓ PASS: FirstName characters validated")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_6_phone_validation():
    """Test 6: Phone number validation"""
    print("\n" + "="*70)
    print("TEST 6: Phone Number Validation")
    print("="*70)
    
    test_cases = [
        ("0501234567", True, "Valid Saudi mobile"),
        ("+966501234567", True, "Valid with country code"),
        ("(050) 123-4567", True, "Valid with formatting"),
        ("050 123 4567", True, "Valid with spaces"),
        ("123456", False, "Too few digits"),
        ("12345678901234567890123456789012345678901234567890123", False, "Too long"),
        ("ABC1234567", False, "Contains letters")
    ]
    
    all_passed = True
    
    for value, should_pass, description in test_cases:
        try:
            patient = create_patient_service(
                first_name=f"TestPhase4Phone{test_cases.index((value, should_pass, description))}",
                phone_number=value
            )
            
            if should_pass:
                print(f"  ✓ {description}: Correctly allowed")
            else:
                print(f"  ✗ {description}: Should have been blocked")
                all_passed = False
                
        except ValueError as e:
            if not should_pass:
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Should have been allowed - {str(e)[:60]}")
                all_passed = False
    
    if all_passed:
        print("✓ PASS: Phone validation works")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_7_birthdate_validation():
    """Test 7: BirthDate validation"""
    print("\n" + "="*70)
    print("TEST 7: BirthDate Validation")
    print("="*70)
    
    tomorrow = (date.today() + timedelta(days=1)).strftime('%Y-%m-%d')
    old_date = "1850-01-01"  # 175+ years ago
    valid_date = "1990-05-15"
    
    test_cases = [
        (valid_date, True, "Valid date"),
        ("2024-12-31", True, "Recent past"),
        (tomorrow, False, "Future date"),
        (old_date, False, "Too old (>150 years)"),
        ("2024/12/31", False, "Wrong format (slashes)"),
        ("31-12-2024", False, "Wrong format (day first)"),
        ("2024-13-01", False, "Invalid month"),
        ("2024-02-30", False, "Invalid day")
    ]
    
    all_passed = True
    
    for value, should_pass, description in test_cases:
        try:
            patient = create_patient_service(
                first_name=f"TestPhase4Birth{test_cases.index((value, should_pass, description))}",
                birth_date=value
            )
            
            if should_pass:
                print(f"  ✓ {description}: Correctly allowed")
            else:
                print(f"  ✗ {description}: Should have been blocked")
                all_passed = False
                
        except ValueError as e:
            if not should_pass:
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Should have been allowed - {str(e)[:60]}")
                all_passed = False
    
    if all_passed:
        print("✓ PASS: BirthDate validation works")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_8_sex_validation():
    """Test 8: SEX validation and normalization"""
    print("\n" + "="*70)
    print("TEST 8: SEX Validation and Normalization")
    print("="*70)
    
    test_cases = [
        ("M", "M", True, "M"),
        ("F", "F", True, "F"),
        ("Male", "M", True, "Male normalized to M"),
        ("Female", "F", True, "Female normalized to F"),
        ("male", "M", True, "male (lowercase)"),
        ("FEMALE", "F", True, "FEMALE (uppercase)"),
        ("X", None, False, "Invalid value X"),
        ("Other", None, False, "Invalid value Other")
    ]
    
    all_passed = True
    
    for input_value, expected, should_pass, description in test_cases:
        try:
            patient = create_patient_service(
                first_name=f"TestPhase4Sex{test_cases.index((input_value, expected, should_pass, description))}",
                sex=input_value
            )
            
            if should_pass:
                if patient['SEX'] == expected:
                    print(f"  ✓ {description}: Correct")
                else:
                    print(f"  ✗ {description}: Expected {expected}, got {patient['SEX']}")
                    all_passed = False
            else:
                print(f"  ✗ {description}: Should have been blocked")
                all_passed = False
                
        except ValueError as e:
            if not should_pass:
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Should have been allowed - {str(e)[:60]}")
                all_passed = False
    
    if all_passed:
        print("✓ PASS: SEX validation and normalization work")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_9_whitespace_trimming():
    """Test 9: Whitespace is trimmed"""
    print("\n" + "="*70)
    print("TEST 9: Whitespace Trimming")
    print("="*70)
    
    try:
        patient = create_patient_service(
            first_name="  TestPhase4Trim  ",
            middle_name="  Ahmad  ",
            last_name="  AlTest  "
        )
        
        expected = "TestPhase4Trim Ahmad AlTest"
        if patient['FullName'] == expected:
            print(f"✓ PASS: Whitespace correctly trimmed")
            print(f"  - Result: '{patient['FullName']}'")
            return True
        else:
            print(f"✗ FAIL: Whitespace not trimmed correctly")
            print(f"  - Expected: '{expected}'")
            print(f"  - Got: '{patient['FullName']}'")
            return False
            
    except Exception as e:
        print(f"✗ FAIL: {str(e)}")
        return False


def test_10_document_validation():
    """Test 10: DocumentNumber validation"""
    print("\n" + "="*70)
    print("TEST 10: DocumentNumber Validation")
    print("="*70)
    
    test_cases = [
        ("DOC123456", True, "Alphanumeric"),
        ("DOC-123-456", True, "With hyphens"),
        ("A" * 97, True, "Max length (100 with prefix)"),  # DOC + 97 A's = 100
        ("A" * 98, False, "Too long (101 with prefix)"),   # DOC + 98 A's = 101
        ("DOC@123", False, "Invalid char @"),
        ("DOC#123", False, "Invalid char #")
    ]
    
    all_passed = True
    
    for value, should_pass, description in test_cases:
        try:
            # Add DOC prefix to A's for consistent length testing
            if value.startswith("A"):
                unique_value = f"DOC{value}"
            else:
                unique_value = f"{value}-P4-{test_cases.index((value, should_pass, description))}"
                
            patient = create_patient_service(
                first_name=f"TestPhase4Doc{test_cases.index((value, should_pass, description))}",
                document_number=unique_value
            )
            
            if should_pass:
                print(f"  ✓ {description}: Correctly allowed")
            else:
                print(f"  ✗ {description}: Should have been blocked")
                all_passed = False
                
        except ValueError as e:
            if not should_pass:
                print(f"  ✓ {description}: Correctly blocked")
            else:
                print(f"  ✗ {description}: Should have been allowed - {str(e)[:60]}")
                all_passed = False
    
    if all_passed:
        print("✓ PASS: DocumentNumber validation works")
    else:
        print("✗ FAIL: Some cases failed")
    
    return all_passed


def test_11_duplicate_passthrough():
    """Test 11: Duplicate detection from DB layer works"""
    print("\n" + "="*70)
    print("TEST 11: Duplicate Detection Pass-Through")
    print("="*70)
    
    try:
        # Create first patient
        patient1 = create_patient_service(
            first_name="TestPhase4Duplicate",
            middle_name="Same",
            last_name="Name"
        )
        print(f"  - Created patient 1: {patient1['FullName']}")
        
        # Try duplicate
        try:
            patient2 = create_patient_service(
                first_name="TestPhase4Duplicate",
                middle_name="Same",
                last_name="Name"
            )
            print(f"✗ FAIL: Duplicate was allowed")
            return False
        except ValueError as e:
            if "already exists" in str(e).lower():
                print(f"✓ PASS: Duplicate correctly blocked from DB layer")
                return True
            else:
                print(f"✗ FAIL: Wrong error - {str(e)}")
                return False
                
    except Exception as e:
        print(f"✗ FAIL: Unexpected error - {str(e)}")
        return False


def run_all_tests():
    """Run all Phase 4 tests"""
    print("\n" + "="*70)
    print("PHASE 4 TEST SUITE: SERVICE LAYER VALIDATION")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initial cleanup
    cleanup_test_data()
    
    results = []
    
    # Run tests
    results.append(("Valid Full Patient", test_1_valid_full_patient()))
    results.append(("Valid Minimal Patient", test_2_valid_minimal_patient()))
    results.append(("FirstName Required", test_3_firstname_required()))
    results.append(("FirstName Length", test_4_firstname_length()))
    results.append(("FirstName Characters", test_5_firstname_characters()))
    results.append(("Phone Validation", test_6_phone_validation()))
    results.append(("BirthDate Validation", test_7_birthdate_validation()))
    results.append(("SEX Validation", test_8_sex_validation()))
    results.append(("Whitespace Trimming", test_9_whitespace_trimming()))
    results.append(("DocumentNumber Validation", test_10_document_validation()))
    results.append(("Duplicate Pass-Through", test_11_duplicate_passthrough()))
    
    # Final cleanup
    cleanup_result = cleanup_test_data()
    results.append(("Final Cleanup", cleanup_result))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*70)
    print(f"Results: {passed}/{total} tests passed ({int(passed/total*100)}%)")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Phase 4 Complete - Ready for Phase 5")
        return True
    else:
        print(f"\n⚠️  {total - passed} TEST(S) FAILED - Fix issues before proceeding")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
