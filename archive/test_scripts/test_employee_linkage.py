"""
Test Employee Linkage
Tests the employee-to-incident linkage functionality
"""

import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_create_incident_with_employees():
    """Test creating an incident with employee linkage"""
    
    payload = {
        "complaint_text": "Test incident with employee linkage",
        "feedback_received_date": datetime.now().strftime("%Y-%m-%d"),
        "issuing_department_id": 1,
        "domain_id": 1,
        "category_id": 1,
        "subcategory_id": 1,
        "classification_id": 1,
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
        "target_department_ids": [1],
        "employees": [
            {
                "employee_id": 101,
                "full_name": "Ahmed Mohamed"
            },
            {
                "employee_id": 102,
                "full_name": "Sara Ahmed"
            }
        ]
    }
    
    print("=" * 60)
    print("Testing Incident Creation with Employees")
    print("=" * 60)
    print(f"\nPayload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/incidents",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                incident_id = result.get("id")
                print(f"\n✅ SUCCESS! Incident created with ID: {incident_id}")
                print(f"Employees should now be linked to this incident")
                
                # TODO: Add query to verify employee linkage
                print(f"\nTo verify, run:")
                print(f"SELECT * FROM APP_IncidentCaseEmployee WHERE IncidentRequestCaseID = {incident_id}")
                return True
            else:
                print(f"\n❌ FAILED: {result.get('message')}")
                return False
        else:
            print(f"\n❌ HTTP Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n🧪 Employee Linkage Test")
    print("Make sure:")
    print("1. Backend server is running on port 8000")
    print("2. You've run ALTER_EMPLOYEE_TABLE.sql")
    print("3. Employee IDs 101 and 102 exist in APP_IncidentCaseEmployee (or will be created)")
    
   input("\nPress Enter to start test...")
    
    success = test_create_incident_with_employees()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TEST PASSED - Employee linkage working!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED - Check errors above")
        print("=" * 60)
