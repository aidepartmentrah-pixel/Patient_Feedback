"""
Test Authentication Protection on Core Write Routers
Tests that all endpoints in insert_router, follow_up_router, and action_items require authentication.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

# Test counters
total_tests = 0
passed_tests = 0
failed_tests = 0

def test_endpoint(method: str, path: str, data: dict = None, description: str = ""):
    """Test an endpoint without authentication - should return 401"""
    global total_tests, passed_tests, failed_tests
    total_tests += 1
    
    try:
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json=data or {})
        elif method == "PUT":
            response = client.put(path, json=data or {})
        elif method == "PATCH":
            response = client.patch(path, json=data or {})
        else:
            print(f"❌ {method} {path} - Unknown HTTP method")
            failed_tests += 1
            return
        
        # Should return 401 Unauthorized
        if response.status_code == 401:
            passed_tests += 1
            print(f"✅ {method} {path} - Protected (401)")
        else:
            failed_tests += 1
            print(f"❌ {method} {path} - NOT PROTECTED! Got {response.status_code}")
            print(f"   Response: {response.json()}")
    except Exception as e:
        failed_tests += 1
        print(f"❌ {method} {path} - ERROR: {str(e)}")

print("=" * 80)
print("TESTING AUTHENTICATION PROTECTION - Core Write Routers")
print("=" * 80)

# ==================== INSERT ROUTER TESTS ====================
print("\n📋 INSERT ROUTER (10 endpoints)")
print("-" * 80)

test_endpoint("POST", "/api/records/add", {
    "complaint_text": "Test",
    "feedback_received_date": "2024-01-01",
    "issuing_department_id": 1,
    "domain_id": 1,
    "category_id": 1,
    "subcategory_id": 1,
    "classification_id": 1,
    "severity_id": 1,
    "stage_id": 1,
    "harm_id": 1,
    "clinical_risk_type_id": 1,
    "feedback_intent_type_id": 1,
    "requires_explanation": False,
    "immediate_action": "Test",
    "taken_action": "Test",
    "patient_name": "Test Patient",
    "is_inpatient": True,
    "source_id": 1
})

test_endpoint("GET", "/api/records/1")
test_endpoint("PUT", "/api/records/1", {
    "complaint_text": "Updated",
    "feedback_received_date": "2024-01-01",
    "issuing_department_id": 1,
    "domain_id": 1,
    "category_id": 1,
    "subcategory_id": 1,
    "classification_id": 1,
    "severity_id": 1,
    "stage_id": 1,
    "harm_id": 1,
    "clinical_risk_type_id": 1,
    "feedback_intent_type_id": 1,
    "requires_explanation": False,
    "immediate_action": "Test",
    "taken_action": "Test",
    "patient_name": "Test",
    "is_inpatient": True,
    "source_id": 1
})
test_endpoint("GET", "/api/records/test")
test_endpoint("GET", "/api/records/search/patients?q=test&limit=10")
test_endpoint("GET", "/api/records/search/doctors?q=test&limit=10")
test_endpoint("GET", "/api/records/search/employees?q=test&limit=10")
test_endpoint("GET", "/api/records/patient/1")
test_endpoint("GET", "/api/records/doctor/1")
test_endpoint("GET", "/api/records/employee/1")

# ==================== FOLLOW-UP ROUTER TESTS ====================
print("\n📋 FOLLOW-UP ROUTER (12 endpoints)")
print("-" * 80)

test_endpoint("POST", "/api/follow-up/actions", {
    "actionTitle": "Test Action",
    "dueDate": "2026-02-01",
    "priority": "medium"
})
test_endpoint("GET", "/api/follow-up/actions")
test_endpoint("GET", "/api/follow-up/actions/1")
test_endpoint("PATCH", "/api/follow-up/actions/1", {
    "priority": "high"
})
test_endpoint("POST", "/api/follow-up/actions/1/complete", {
    "completionNotes": "Done"
})
test_endpoint("POST", "/api/follow-up/actions/1/delay", {
    "delayDays": 7,
    "reason": "Need approval"
})
test_endpoint("POST", "/api/follow-up/actions/1/reopen", {
    "reopenReason": "Mistake",
    "newDueDate": "2026-02-15"
})
test_endpoint("GET", "/api/follow-up/actions/1/history")
test_endpoint("GET", "/api/follow-up/calendar?year=2026&month=1")
test_endpoint("POST", "/api/follow-up/actions/bulk-complete", {
    "actionIds": [1, 2, 3]
})
test_endpoint("POST", "/api/follow-up/actions/bulk-delay", {
    "actionIds": [1, 2],
    "delayDays": 5
})
test_endpoint("POST", "/api/follow-up/actions/bulk-update", {
    "actionIds": [1, 2],
    "priority": "high"
})

# ==================== ACTION ITEMS ROUTER TESTS ====================
print("\n📋 ACTION ITEMS ROUTER (5 endpoints)")
print("-" * 80)

test_endpoint("GET", "/api/action-items/1")
test_endpoint("GET", "/api/action-items/by-incident/1")
test_endpoint("GET", "/api/action-items/by-seasonal-report/1")
test_endpoint("GET", "/api/action-items/by-season/1")
test_endpoint("POST", "/api/action-items/1/mark-done")

# ==================== SUMMARY ====================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {passed_tests}")
print(f"❌ Failed: {failed_tests}")
print(f"Success Rate: {(passed_tests/total_tests*100):.1f}%")
print("=" * 80)

if failed_tests == 0:
    print("\n🎉 ALL TESTS PASSED! All endpoints are properly protected.")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed_tests} tests failed. Please review the output above.")
    sys.exit(1)
