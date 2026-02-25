"""
Test Employee Validation with HR System
Tests that employees are properly validated against APP_VIEWTABLE_HR_EMPLOYEES
"""

import requests
import json
from datetime import datetime
from backend.core.database import get_connection

BASE_URL = "http://localhost:8000"
session = requests.Session()


def login():
    """Login to get authenticated session"""
    response = session.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "software_admin", "password": "admin123"}
    )
    if response.status_code == 200:
        print("✅ Logged in successfully\n")
        return True
    else:
        print(f"❌ Login failed: {response.status_code}")
        return False


def get_real_employees():
    """Get real employee IDs from HR system"""
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT TOP 3
                EmployeeID,
                FullName,
                JobTitle
            FROM APP_VIEWTABLE_HR_EMPLOYEES
            WHERE IsActive = 1
            ORDER BY EmployeeID
        """)
        
        employees = []
        for row in cursor.fetchall():
            employees.append({
                "employee_id": row.EmployeeID,
                "full_name": row.FullName,
                "job_title": row.JobTitle
            })
        
        return employees
        
    except Exception as e:
        print(f"Error fetching employees: {e}")
        return []
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def test_valid_employees():
    """Test creating incident with valid employee IDs"""
    
    print("=" * 70)
    print("TEST 1: Valid Employee IDs from HR System")
    print("=" * 70)
    
    # Get real employees from HR system
    real_employees = get_real_employees()
    
    if not real_employees:
        print("❌ Could not fetch real employees from HR system")
        return False
    
    print("\n📋 Using REAL employees from APP_VIEWTABLE_HR_EMPLOYEES:")
    for emp in real_employees[:2]:
        print(f"   - Employee {emp['employee_id']}: {emp['full_name']}")
    
    payload = {
        "complaint_text": "Test with REAL HR employees",
        "feedback_received_date": datetime.now().strftime("%Y-%m-%d"),
        "issuing_department_id": 43,
        "domain_id": 1,
        "category_id": 6,
        "subcategory_id": 19,
        "classification_id": 132,
        "severity_id": 1,
        "stage_id": 1,
        "harm_id": 1,
        "requires_explanation": True,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "Test immediate action",
        "taken_action": "Test taken action",
        "patient_name": "Test Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 1,
        "target_department_ids": [43],
        "employees": [
            {
                "employee_id": real_employees[0]['employee_id'],
                "full_name": "WRONG NAME"  # Should be ignored, replaced with HR name
            },
            {
                "employee_id": real_employees[1]['employee_id'],
                "full_name": "FAKE NAME"  # Should be ignored, replaced with HR name
            }
        ]
    }
    
    print("\n🔄 Sending request...")
    print(f"   Note: Sending WRONG names to test that backend fetches correct names from HR")
    
    try:
        response = session.post(
            f"{BASE_URL}/api/records/add",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                incident_id = result.get("id")
                print(f"\n✅ SUCCESS! Incident {incident_id} created")
                return incident_id
            else:
                print(f"\n❌ FAILED: {result.get('message')}")
                return None
        else:
            print(f"\n❌ HTTP Error {response.status_code}: {response.json()}")
            return None
            
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}")
        return None


def test_invalid_employee():
    """Test that invalid employee IDs are rejected"""
    
    print("\n" + "=" * 70)
    print("TEST 2: Invalid Employee ID (Should Be Rejected)")
    print("=" * 70)
    
    payload = {
        "complaint_text": "Test with INVALID employee",
        "feedback_received_date": datetime.now().strftime("%Y-%m-%d"),
        "issuing_department_id": 43,
        "domain_id": 1,
        "category_id": 6,
        "subcategory_id": 19,
        "classification_id": 132,
        "severity_id": 1,
        "stage_id": 1,
        "harm_id": 1,
        "requires_explanation": True,
        "clinical_risk_type_id": 1,
        "feedback_intent_type_id": 1,
        "immediate_action": "Test immediate action",
        "taken_action": "Test taken action",
        "patient_name": "Test Patient",
        "is_inpatient": True,
        "source_id": 1,
        "building_id": 1,
        "target_department_ids": [43],
        "employees": [
            {
                "employee_id": 999999,  # Invalid employee ID
                "full_name": "Fake Employee"
            }
        ]
    }
    
    print("\n🔄 Sending request with Employee ID 999999 (should not exist)...")
    
    try:
        response = session.post(
            f"{BASE_URL}/api/records/add",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 400:
            error = response.json().get('detail', {})
            if error.get('error') == 'INVALID_REFERENCE' and 'Employee' in error.get('message', ''):
                print(f"\n✅ CORRECTLY REJECTED!")
                print(f"   Error: {error.get('message')}")
                return True
            else:
                print(f"\n❌ Wrong error type: {error}")
                return False
        else:
            print(f"\n❌ Should have been rejected but got: {response.status_code}")
            print(response.json())
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}")
        return False


def verify_correct_names(incident_id):
    """Verify that employee names in DB match HR system, not frontend payload"""
    
    print("\n" + "=" * 70)
    print("TEST 3: Verify Correct Names Stored from HR System")
    print("=" * 70)
    
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get employees linked to this incident
        cursor.execute("""
            SELECT 
                e.EmployeeID,
                e.FullName AS StoredName,
                hr.FullName AS ActualHRName
            FROM dbo.APP_IncidentCaseEmployee e
            LEFT JOIN dbo.APP_VIEWTABLE_HR_EMPLOYEES hr ON e.EmployeeID = hr.EmployeeID
            WHERE e.IncidentRequestCaseID = ?
            ORDER BY e.EmployeeID
        """, (incident_id,))
        
        employees = cursor.fetchall()
        
        if not employees:
            print(f"\n❌ No employees found for incident {incident_id}")
            return False
        
        print(f"\n📊 Comparing stored names vs HR system names:")
        print("-" * 70)
        
        all_match = True
        for emp in employees:
            match_status = "✅ MATCH" if emp.StoredName == emp.ActualHRName else "❌ MISMATCH"
            print(f"\nEmployee {emp.EmployeeID}:")
            print(f"  Stored in DB:    {emp.StoredName}")
            print(f"  Actual in HR:    {emp.ActualHRName}")
            print(f"  Status:          {match_status}")
            
            if emp.StoredName != emp.ActualHRName:
                all_match = False
        
        print("-" * 70)
        
        if all_match:
            print("\n✅ ALL NAMES MATCH - Backend correctly fetched from HR system!")
            return True
        else:
            print("\n❌ NAME MISMATCH - Backend is not fetching from HR system!")
            return False
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🧪 EMPLOYEE HR VALIDATION TEST SUITE")
    print("=" * 70)
    print("\nThis test verifies:")
    print("1. Only valid HR employee IDs are accepted")
    print("2. Invalid employee IDs are rejected")
    print("3. Employee names are fetched from HR system, not frontend")
    print("\n" + "=" * 70)
    
    input("\nPress Enter to start tests...")
    
    # Login
    if not login():
        print("\n❌ Cannot proceed without authentication")
        exit(1)
    
    # Test 1: Valid employees
    incident_id = test_valid_employees()
    
    # Test 2: Invalid employee
    invalid_rejected = test_invalid_employee()
    
    # Test 3: Verify correct names were stored
    names_correct = False
    if incident_id:
        names_correct = verify_correct_names(incident_id)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Valid employees accepted:     {'PASS' if incident_id else 'FAIL'}")
    print(f"✅ Invalid employees rejected:   {'PASS' if invalid_rejected else 'FAIL'}")
    print(f"✅ HR names correctly stored:    {'PASS' if names_correct else 'FAIL'}")
    print("=" * 70)
    
    if incident_id and invalid_rejected and names_correct:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Employee validation is working correctly!")
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("⚠️  Employee validation needs fixes!")
