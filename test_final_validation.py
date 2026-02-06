"""
FINAL VALIDATION TEST - Authentication Protection
Comprehensive validation that all 27 endpoints are properly protected.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

print("=" * 90)
print(" FINAL VALIDATION: AUTHENTICATION PROTECTION ON CORE WRITE ROUTERS")
print("=" * 90)

# All 27 endpoints that should be protected
protected_endpoints = [
    # ==================== INSERT ROUTER (10 endpoints) ====================
    {
        "router": "insert_router",
        "method": "POST",
        "path": "/api/records/add",
        "description": "Create new record",
        "data": {
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
            "patient_name": "Test",
            "is_inpatient": True,
            "source_id": 1
        }
    },
    {"router": "insert_router", "method": "GET", "path": "/api/records/1", "description": "Get record by ID"},
    {"router": "insert_router", "method": "PUT", "path": "/api/records/1", "description": "Update record", "data": {"complaint_text": "Updated"}},
    {"router": "insert_router", "method": "GET", "path": "/api/records/test", "description": "Test endpoint"},
    {"router": "insert_router", "method": "GET", "path": "/api/records/search/patients?q=test", "description": "Search patients"},
    {"router": "insert_router", "method": "GET", "path": "/api/records/search/doctors?q=test", "description": "Search doctors"},
    {"router": "insert_router", "method": "GET", "path": "/api/records/search/employees?q=test", "description": "Search employees"},
    {"router": "insert_router", "method": "GET", "path": "/api/records/patient/1", "description": "Get patient by ID"},
    {"router": "insert_router", "method": "GET", "path": "/api/records/doctor/1", "description": "Get doctor by ID"},
    {"router": "insert_router", "method": "GET", "path": "/api/records/employee/1", "description": "Get employee by ID"},
    
    # ==================== FOLLOW-UP ROUTER (12 endpoints) ====================
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions", "description": "Create action", "data": {"actionTitle": "Test", "dueDate": "2026-02-01", "priority": "medium"}},
    {"router": "follow_up_router", "method": "GET", "path": "/api/follow-up/actions", "description": "List actions"},
    {"router": "follow_up_router", "method": "GET", "path": "/api/follow-up/actions/1", "description": "Get action by ID"},
    {"router": "follow_up_router", "method": "PATCH", "path": "/api/follow-up/actions/1", "description": "Update action", "data": {"priority": "high"}},
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions/1/complete", "description": "Complete action", "data": {}},
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions/1/delay", "description": "Delay action", "data": {"delayDays": 7}},
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions/1/reopen", "description": "Reopen action", "data": {"reopenReason": "Test"}},
    {"router": "follow_up_router", "method": "GET", "path": "/api/follow-up/actions/1/history", "description": "Get action history"},
    {"router": "follow_up_router", "method": "GET", "path": "/api/follow-up/calendar?year=2026&month=1", "description": "Calendar view"},
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions/bulk-complete", "description": "Bulk complete", "data": {"actionIds": [1, 2]}},
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions/bulk-delay", "description": "Bulk delay", "data": {"actionIds": [1], "delayDays": 5}},
    {"router": "follow_up_router", "method": "POST", "path": "/api/follow-up/actions/bulk-update", "description": "Bulk update", "data": {"actionIds": [1], "priority": "high"}},
    
    # ==================== ACTION ITEMS ROUTER (5 endpoints) ====================
    {"router": "action_items", "method": "GET", "path": "/api/action-items/1", "description": "Get action item by ID"},
    {"router": "action_items", "method": "GET", "path": "/api/action-items/by-incident/1", "description": "Get by incident"},
    {"router": "action_items", "method": "GET", "path": "/api/action-items/by-seasonal-report/1", "description": "Get by seasonal report"},
    {"router": "action_items", "method": "GET", "path": "/api/action-items/by-season/1", "description": "Get by season"},
    {"router": "action_items", "method": "POST", "path": "/api/action-items/1/mark-done", "description": "Mark action done", "data": {}},
]

# Counters
total = len(protected_endpoints)
passed = 0
failed = 0
failed_endpoints = []

# Group by router
router_stats = {
    "insert_router": {"total": 0, "passed": 0, "failed": 0},
    "follow_up_router": {"total": 0, "passed": 0, "failed": 0},
    "action_items": {"total": 0, "passed": 0, "failed": 0}
}

print(f"\n📊 Testing {total} protected endpoints...\n")

for endpoint in protected_endpoints:
    router = endpoint["router"]
    method = endpoint["method"]
    path = endpoint["path"]
    desc = endpoint["description"]
    data = endpoint.get("data")
    
    router_stats[router]["total"] += 1
    
    try:
        # Make request
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json=data)
        elif method == "PUT":
            response = client.put(path, json=data)
        elif method == "PATCH":
            response = client.patch(path, json=data)
        else:
            response = None
        
        # Check result
        if response and response.status_code == 401:
            passed += 1
            router_stats[router]["passed"] += 1
            print(f"✅ {method:6} {path:55} (Protected)")
        else:
            failed += 1
            router_stats[router]["failed"] += 1
            failed_endpoints.append({
                "method": method,
                "path": path,
                "description": desc,
                "status": response.status_code if response else "No response"
            })
            print(f"❌ {method:6} {path:55} (Status: {response.status_code if response else 'Error'})")
            
    except Exception as e:
        failed += 1
        router_stats[router]["failed"] += 1
        failed_endpoints.append({
            "method": method,
            "path": path,
            "description": desc,
            "error": str(e)
        })
        print(f"❌ {method:6} {path:55} (ERROR: {str(e)[:30]})")

# Print router summaries
print("\n" + "=" * 90)
print(" ROUTER SUMMARIES")
print("=" * 90)

for router_name, stats in router_stats.items():
    total_router = stats["total"]
    passed_router = stats["passed"]
    failed_router = stats["failed"]
    percentage = (passed_router / total_router * 100) if total_router > 0 else 0
    
    icon = "✅" if failed_router == 0 else "❌"
    print(f"\n{icon} {router_name:20} → {passed_router}/{total_router} passed ({percentage:.0f}%)")

# Print overall summary
print("\n" + "=" * 90)
print(" FINAL RESULTS")
print("=" * 90)
print(f"\n📊 Total Endpoints Tested: {total}")
print(f"✅ Passed (401): {passed}")
print(f"❌ Failed: {failed}")
print(f"📈 Success Rate: {(passed/total*100):.1f}%")

if failed == 0:
    print("\n" + "🎉" * 30)
    print("\n✅ ✅ ✅  100% SUCCESS - ALL ENDPOINTS PROTECTED  ✅ ✅ ✅")
    print("\n" + "🎉" * 30)
    print("\n✓ All 10 insert_router endpoints return 401 without auth")
    print("✓ All 12 follow_up_router endpoints return 401 without auth")
    print("✓ All 5 action_items endpoints return 401 without auth")
    print("\n✓ Total: 27 endpoints properly protected")
    print("✓ Guards (require_logged_in) working correctly")
    print("✓ Dependencies (get_current_user) working correctly")
    print("\n" + "=" * 90)
    sys.exit(0)
else:
    print("\n" + "=" * 90)
    print(f" ⚠️  {failed} ENDPOINTS FAILED")
    print("=" * 90)
    for ep in failed_endpoints:
        print(f"\n❌ {ep['method']} {ep['path']}")
        print(f"   Description: {ep['description']}")
        if 'status' in ep:
            print(f"   Status: {ep['status']}")
        if 'error' in ep:
            print(f"   Error: {ep['error']}")
    print("\n" + "=" * 90)
    sys.exit(1)
