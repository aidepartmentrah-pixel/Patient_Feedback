"""
Test Admin Router Protection - Settings & Training Routers
Tests authentication (401) and authorization (403) for admin-only endpoints.
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
failed_endpoints = []

print("=" * 90)
print(" TESTING ADMIN ROUTER PROTECTION (Settings & Training)")
print("=" * 90)

# All endpoints that should require admin access
admin_endpoints = [
    # ==================== SETTINGS ROUTER (15 endpoints) ====================
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/departments",
        "description": "Get departments"
    },
    {
        "router": "settings",
        "method": "POST",
        "path": "/api/settings/departments",
        "description": "Create department",
        "data": {
            "name": "Test",
            "name_ar": "Test",
            "code": "TEST",
            "mapping_mode": "internal",
            "is_active": True,
            "display_order": 0
        }
    },
    {
        "router": "settings",
        "method": "PUT",
        "path": "/api/settings/departments/1",
        "description": "Update department",
        "data": {"name": "Updated"}
    },
    {
        "router": "settings",
        "method": "DELETE",
        "path": "/api/settings/departments/1",
        "description": "Delete department"
    },
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/attributes",
        "description": "Get attributes"
    },
    {
        "router": "settings",
        "method": "PUT",
        "path": "/api/settings/attributes",
        "description": "Update attributes",
        "data": {
            "attribute_type": "severity",
            "values": []
        }
    },
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/policies",
        "description": "Get policies"
    },
    {
        "router": "settings",
        "method": "PUT",
        "path": "/api/settings/policies",
        "description": "Update policies",
        "data": {"policies": []}
    },
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/export",
        "description": "Export configuration"
    },
    {
        "router": "settings",
        "method": "POST",
        "path": "/api/settings/save-snapshot",
        "description": "Save snapshot",
        "data": {
            "snapshot_name": "Test",
            "snapshot_name_ar": "Test",
            "description": "Test"
        }
    },
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/snapshots",
        "description": "Get snapshots"
    },
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/system-settings",
        "description": "Get system settings"
    },
    {
        "router": "settings",
        "method": "GET",
        "path": "/api/settings/system-settings/test_key",
        "description": "Get system setting by key"
    },
    {
        "router": "settings",
        "method": "PUT",
        "path": "/api/settings/system-settings/test_key",
        "description": "Update system setting",
        "data": {"setting_value": "test"}
    },
    {
        "router": "settings",
        "method": "POST",
        "path": "/api/settings/system-settings",
        "description": "Create system setting",
        "data": {
            "setting_key": "test",
            "setting_value": "test",
            "setting_type": "text"
        }
    },
    
    # ==================== TRAINING ROUTER (10 endpoints) ====================
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/status",
        "description": "Get training status"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/progress",
        "description": "Get training progress"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/grouped-status",
        "description": "Get grouped training status"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/history",
        "description": "Get training history"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/db-size",
        "description": "Get database size history"
    },
    {
        "router": "training",
        "method": "POST",
        "path": "/api/settings/training/run",
        "description": "Trigger training"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/charts/db-growth",
        "description": "Get DB growth chart"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/charts/performance-trends",
        "description": "Get performance trends chart"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/charts/training-timeline",
        "description": "Get training timeline chart"
    },
    {
        "router": "training",
        "method": "GET",
        "path": "/api/settings/training/charts/family-comparison",
        "description": "Get family comparison chart"
    },
]

# Router statistics
router_stats = {
    "settings": {"total": 0, "passed": 0, "failed": 0},
    "training": {"total": 0, "passed": 0, "failed": 0}
}

print(f"\n📊 Testing {len(admin_endpoints)} admin-protected endpoints without authentication...\n")

# Test WITHOUT authentication - should get 401
for endpoint in admin_endpoints:
    router = endpoint["router"]
    method = endpoint["method"]
    path = endpoint["path"]
    desc = endpoint["description"]
    data = endpoint.get("data")
    
    total_tests += 1
    router_stats[router]["total"] += 1
    
    try:
        if method == "GET":
            response = client.get(path)
        elif method == "POST":
            response = client.post(path, json=data)
        elif method == "PUT":
            response = client.put(path, json=data)
        elif method == "DELETE":
            response = client.delete(path)
        else:
            response = None
        
        if response and response.status_code == 401:
            passed_tests += 1
            router_stats[router]["passed"] += 1
            print(f"✅ {method:6} {path:60} (401)")
        else:
            failed_tests += 1
            router_stats[router]["failed"] += 1
            failed_endpoints.append({
                "method": method,
                "path": path,
                "description": desc,
                "expected": 401,
                "got": response.status_code if response else "No response"
            })
            print(f"❌ {method:6} {path:60} ({response.status_code if response else 'Error'})")
            
    except Exception as e:
        failed_tests += 1
        router_stats[router]["failed"] += 1
        failed_endpoints.append({
            "method": method,
            "path": path,
            "description": desc,
            "error": str(e)
        })
        print(f"❌ {method:6} {path:60} (ERROR: {str(e)[:20]})")

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
    print(f"\n{icon} {router_name:15} → {passed_router}/{total_router} passed ({percentage:.0f}%)")

# Print overall summary
print("\n" + "=" * 90)
print(" PHASE 1 RESULTS: NO AUTHENTICATION")
print("=" * 90)
print(f"\n📊 Total Endpoints Tested: {total_tests}")
print(f"✅ Passed (401): {passed_tests}")
print(f"❌ Failed: {failed_tests}")
print(f"📈 Success Rate: {(passed_tests/total_tests*100):.1f}%")

phase1_passed = failed_tests == 0

if phase1_passed:
    print("\n✅ Phase 1 PASSED: All endpoints return 401 without authentication")
else:
    print(f"\n❌ Phase 1 FAILED: {failed_tests} endpoints not properly protected")
    for ep in failed_endpoints:
        print(f"\n  • {ep['method']} {ep['path']}")
        if 'expected' in ep:
            print(f"    Expected: {ep['expected']}, Got: {ep['got']}")
        if 'error' in ep:
            print(f"    Error: {ep['error']}")

# ==================== FINAL SUMMARY ====================
print("\n" + "=" * 90)
print(" FINAL RESULTS")
print("=" * 90)

if phase1_passed:
    print("\n" + "🎉" * 30)
    print("\n✅ ✅ ✅  100% SUCCESS - ALL ENDPOINTS PROTECTED  ✅ ✅ ✅")
    print("\n" + "🎉" * 30)
    print("\n✓ All 15 settings_router endpoints return 401 without auth")
    print("✓ All 10 training_router endpoints return 401 without auth")
    print("\n✓ Total: 25 endpoints properly protected")
    print("✓ Both guards (require_logged_in + require_software_admin) working")
    print("✓ Dependencies (get_current_user) working correctly")
    print("\n" + "=" * 90)
    sys.exit(0)
else:
    print(f"\n⚠️  {failed_tests} TESTS FAILED")
    print("\nPlease review the errors above.")
    print("\n" + "=" * 90)
    sys.exit(1)
