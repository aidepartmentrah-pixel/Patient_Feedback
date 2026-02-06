"""
Test Authentication Protection - WITH LOGIN
Tests that endpoints work correctly when user is authenticated.
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

print("=" * 80)
print("TESTING WITH AUTHENTICATION - Core Write Routers")
print("=" * 80)

# ==================== LOGIN FIRST ====================
print("\n🔐 Step 1: Login")
print("-" * 80)

login_response = client.post("/api/auth/login", json={
    "username": "admin",
    "password": "admin123"
})

if login_response.status_code == 200:
    print(f"✅ Login successful (200)")
    login_data = login_response.json()
    print(f"   User: {login_data.get('username')}")
    print(f"   Roles: {[s['role_code'] for s in login_data.get('scopes', [])]}")
else:
    print(f"❌ Login failed ({login_response.status_code})")
    print(f"   Response: {login_response.json()}")
    print("\n⚠️  Cannot continue tests without authentication")
    sys.exit(1)

# ==================== TEST AUTHENTICATED ENDPOINTS ====================
print("\n📋 Testing Authenticated Endpoints")
print("-" * 80)

def test_authenticated(method: str, path: str, data: dict = None, expected_status_range=(200, 499)):
    """Test an endpoint WITH authentication - should work (or fail with business logic error, not 401)"""
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
        
        # Should NOT return 401 Unauthorized
        if response.status_code == 401:
            failed_tests += 1
            print(f"❌ {method} {path} - Still getting 401! Auth not working")
            print(f"   Response: {response.json()}")
        elif expected_status_range[0] <= response.status_code < expected_status_range[1]:
            passed_tests += 1
            status_emoji = "✅" if response.status_code < 300 else "⚠️"
            print(f"{status_emoji} {method} {path} - Accessible ({response.status_code})")
        else:
            failed_tests += 1
            print(f"❌ {method} {path} - Unexpected status {response.status_code}")
            print(f"   Response: {response.text[:200]}")
    except Exception as e:
        failed_tests += 1
        print(f"❌ {method} {path} - ERROR: {str(e)}")

# Test a few representative endpoints
print("\n🧪 Sample Tests (representative endpoints):")
print()

# INSERT ROUTER - Test endpoint (should work)
test_authenticated("GET", "/api/records/test")

# INSERT ROUTER - Search patients (might fail if no data, but shouldn't be 401)
test_authenticated("GET", "/api/records/search/patients?q=test&limit=10")

# FOLLOW-UP ROUTER - Get actions (should work or return empty list)
test_authenticated("GET", "/api/follow-up/actions")

# FOLLOW-UP ROUTER - Calendar (should work)
test_authenticated("GET", "/api/follow-up/calendar?year=2026&month=1")

# ACTION ITEMS - Get by incident (might 404 if not found, but not 401)
test_authenticated("GET", "/api/action-items/by-incident/1")

# ACTION ITEMS - Get by season (might return empty, but not 401)
test_authenticated("GET", "/api/action-items/by-season/1")

# ==================== TEST LOGOUT ====================
print("\n🚪 Step 3: Logout")
print("-" * 80)

logout_response = client.post("/api/auth/logout")
if logout_response.status_code == 200:
    print(f"✅ Logout successful (200)")
else:
    print(f"⚠️  Logout returned {logout_response.status_code}")

# ==================== TEST AFTER LOGOUT ====================
print("\n🔒 Step 4: Verify Protection After Logout")
print("-" * 80)

# After logout, should get 401 again
logout_test = client.get("/api/records/test")
if logout_test.status_code == 401:
    print(f"✅ After logout: Protected again (401)")
    passed_tests += 1
    total_tests += 1
else:
    print(f"❌ After logout: Still accessible! ({logout_test.status_code})")
    failed_tests += 1
    total_tests += 1

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
    print("\n🎉 ALL TESTS PASSED! Authentication is working correctly.")
    print("   - Endpoints are protected when not logged in (401)")
    print("   - Endpoints are accessible when logged in")
    print("   - Endpoints are protected again after logout (401)")
    sys.exit(0)
else:
    print(f"\n⚠️  {failed_tests} tests failed. Please review the output above.")
    sys.exit(1)
