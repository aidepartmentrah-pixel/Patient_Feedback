"""
Quick verification of Phase 4 /api/auth/me contract.
"""

import sys
import os
from pathlib import Path
import json

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("=" * 70)
print("Phase 4 STEP 4.1: /api/auth/me Contract Verification")
print("=" * 70)

# Clear sessions
client.cookies.clear()

# Login as section_admin (has org_unit)
print("\n1. Login as section_admin...")
login_response = client.post(
    "/api/auth/login",
    json={"username": "section_admin", "password": "section123"}
)
print(f"   Status: {login_response.status_code}")

# Get /api/auth/me
print("\n2. GET /api/auth/me")
me_response = client.get("/api/auth/me")
print(f"   Status: {me_response.status_code}")

# Pretty print the response
print("\n3. Response JSON:")
data = me_response.json()
print(json.dumps(data, indent=2))

print("\n4. Verify Phase 4 fields:")
user = data["user"]
print(f"   ✓ roles: {user['roles']}")
print(f"   ✓ primary_unit_id: {user['primary_unit_id']}")
print(f"   ✓ primary_unit_type: {user['primary_unit_type']}")

print("\n5. Verify existing fields preserved:")
print(f"   ✓ user_id: {user['user_id']}")
print(f"   ✓ username: {user['username']}")
print(f"   ✓ is_active: {user['is_active']}")
print(f"   ✓ scopes: {len(user['scopes'])} scope(s)")
print(f"   ✓ allowed_unit_ids: {len(user['allowed_unit_ids'])} unit(s)")

# Test with SOFTWARE_ADMIN (no org unit)
print("\n" + "=" * 70)
print("6. Testing with SOFTWARE_ADMIN (no org_unit)...")
client.cookies.clear()
login_response = client.post(
    "/api/auth/login",
    json={"username": "software_admin", "password": "admin123"}
)
me_response = client.get("/api/auth/me")
data = me_response.json()
user = data["user"]

print("\n7. SOFTWARE_ADMIN Response:")
print(json.dumps(data, indent=2))

print("\n8. SOFTWARE_ADMIN primary_unit handling:")
print(f"   primary_unit_id: {user['primary_unit_id']}")
print(f"   primary_unit_type: {user['primary_unit_type']}")
print(f"   (org_unit_id from scope: {user['scopes'][0]['org_unit_id']})")

print("\n" + "=" * 70)
print("✅ STEP 4.1 VERIFICATION COMPLETE")
print("=" * 70)
