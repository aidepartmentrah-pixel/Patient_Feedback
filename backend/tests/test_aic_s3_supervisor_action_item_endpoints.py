"""
Endpoint Test Suite for Action Item Coordination (Iteration 4) — Session 3
Hits the LIVE backend over HTTP (http://127.0.0.1:8000) to verify:
- RBAC enforcement (only Complaint Supervisor/Software Admin can create/cancel)
- Scope enforcement (list/complete respect the caller's org-unit scope)
- Validation failures (missing fields, bad case, frozen unit)
- Full lifecycle (create -> list -> complete / cancel)

Requires the backend to be running (Start-Backend.ps1) and the standard
test accounts to exist (see feedback_credentials memory).

Run: python backend/tests/test_aic_s3_supervisor_action_item_endpoints.py
"""

import sys
sys.path.insert(0, 'backend')

import requests

BASE_URL = "http://127.0.0.1:8000"

CASE_ID = 1
FROZEN_UNIT = 12
VALID_UNIT = 1     # الادارة العامة -- administration_admin (UserID=6) is scoped here
OUT_OF_SCOPE_UNIT = 2   # الادارة المالية -- a separate root, NOT in section_admin's scope

tests_run = 0
tests_passed = 0
tests_failed = 0


def test(description):
    def decorator(func):
        def wrapper():
            global tests_run, tests_passed, tests_failed
            tests_run += 1
            try:
                func()
                tests_passed += 1
                print(f"PASS: {description}")
                return True
            except AssertionError as e:
                tests_failed += 1
                print(f"FAIL: {description}")
                print(f"   Error: {e}")
                return False
            except Exception as e:
                tests_failed += 1
                print(f"ERROR: {description}")
                print(f"   Exception: {e}")
                return False
        return wrapper
    return decorator


def login(username, password):
    s = requests.Session()
    resp = s.post(f"{BASE_URL}/api/auth/login", json={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed for {username}: {resp.status_code} {resp.text}"
    return s


print("=" * 70)
print("AIC-S3: SUPERVISOR ACTION ITEM ENDPOINT TEST SUITE (live HTTP)")
print("=" * 70)
print()

print("Logging in test accounts...")
supervisor = login("complaint_supervisor", "5bb5a339")
section_admin = login("section_admin", "section2026")
department_admin = login("department_admin", "department2026")
administration_admin = login("administration_admin", "admin2026")
print("All accounts logged in.")
print()

created_item_ids = []


@test("1. Non-supervisor (section_admin) gets 403 creating an action item")
def t1():
    resp = section_admin.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": VALID_UNIT,
        "description": "Should be forbidden",
    })
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"

t1()


@test("2. Complaint Supervisor gets 400 when description is missing")
def t2():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": VALID_UNIT,
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"

t2()


@test("3. Complaint Supervisor gets 404 targeting a nonexistent case")
def t3():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": -999999,
        "target_org_unit_id": VALID_UNIT,
        "description": "Should 404 - bad case",
    })
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"

t3()


@test("4. Complaint Supervisor gets 400 targeting a frozen org unit")
def t4():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": FROZEN_UNIT,
        "description": "Should 400 - frozen unit",
    })
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"

t4()


@test("5. Complaint Supervisor creates a valid action item -> 200/201, status PENDING")
def t5():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": VALID_UNIT,
        "description": "AIC-S3 endpoint test item (targets Finance Administration)",
        "due_date": "2026-12-31",
    })
    assert resp.status_code in (200, 201), f"Expected 200/201, got {resp.status_code}: {resp.text}"
    body = resp.json()
    item = body["item"]
    assert item["status"] == "PENDING"
    assert item["target_org_unit_id"] == VALID_UNIT
    created_item_ids.append(item["action_item_id"])
    global created_item_id
    created_item_id = item["action_item_id"]

t5()


@test("6. administration_admin (scoped to unit 1 - الادارة العامة) sees the item in list")
def t6():
    resp = administration_admin.get(f"{BASE_URL}/api/v2/supervisor-action-items")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    items = resp.json()["items"]
    assert any(i["action_item_id"] == created_item_id for i in items), \
        "administration_admin (unit 1 scope) should see the item targeted at unit 1"

t6()


@test("7. section_admin (unit 10, separate root) does NOT see the item in list")
def t7():
    resp = section_admin.get(f"{BASE_URL}/api/v2/supervisor-action-items")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    items = resp.json()["items"]
    assert not any(i["action_item_id"] == created_item_id for i in items), \
        "section_admin (unit 10 scope) should NOT see an item targeted at unit 1"

t7()


@test("8. section_admin (out-of-scope) gets 403 trying to complete the item")
def t8():
    resp = section_admin.post(f"{BASE_URL}/api/v2/supervisor-action-items/{created_item_id}/complete")
    assert resp.status_code == 403, f"Expected 403, got {resp.status_code}: {resp.text}"

t8()


@test("9. administration_admin (in scope for unit 1) completes the item -> 200, COMPLETED")
def t9():
    resp = administration_admin.post(f"{BASE_URL}/api/v2/supervisor-action-items/{created_item_id}/complete")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    item = resp.json()["item"]
    assert item["status"] == "COMPLETED"

t9()


@test("10. Completing the same item again is rejected (already COMPLETED)")
def t10():
    resp = administration_admin.post(f"{BASE_URL}/api/v2/supervisor-action-items/{created_item_id}/complete")
    assert resp.status_code == 400, f"Expected 400, got {resp.status_code}: {resp.text}"

t10()


@test("11. Non-creator (administration_admin) gets 403 cancelling a PENDING item")
def t11():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": VALID_UNIT,
        "description": "AIC-S3 endpoint test item (for cancel-by-non-creator test)",
    })
    item_id = resp.json()["item"]["action_item_id"]
    created_item_ids.append(item_id)

    resp2 = administration_admin.post(f"{BASE_URL}/api/v2/supervisor-action-items/{item_id}/cancel")
    assert resp2.status_code == 403, f"Expected 403, got {resp2.status_code}: {resp2.text}"

t11()


@test("12. Creator (complaint_supervisor) cancels their own item -> 200, status CANCELLED")
def t12():
    item_id = created_item_ids[-1]
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items/{item_id}/cancel")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    item = resp.json()["item"]
    assert item["status"] == "CANCELLED"

t12()


@test("13. Completing/cancelling a nonexistent action item returns 404")
def t13():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items/-999999/complete")
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}: {resp.text}"
    resp2 = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items/-999999/cancel")
    assert resp2.status_code == 404, f"Expected 404, got {resp2.status_code}: {resp2.text}"

t13()


# ============================================================
# CLEANUP
# ============================================================

print()
print("--- CLEANUP: removing test action items via direct SQL ---")
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()
cursor.execute("SET QUOTED_IDENTIFIER ON")
if created_item_ids:
    placeholders = ','.join(['?'] * len(created_item_ids))
    cursor.execute(
        f"DELETE FROM dbo.APP_SupervisorActionItemAuditLog WHERE ActionItemID IN ({placeholders})",
        created_item_ids
    )
    cursor.execute(
        f"DELETE FROM dbo.APP_SupervisorActionItem WHERE ActionItemID IN ({placeholders})",
        created_item_ids
    )
    conn.commit()
cursor.close()
conn.close()
print(f"Cleaned up {len(created_item_ids)} test action item(s).")

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests:  {tests_run}")
print(f"Passed:       {tests_passed}")
print(f"Failed:       {tests_failed}")
print()

if tests_failed == 0:
    print("ALL TESTS PASSED!")
    exit(0)
else:
    print(f"{tests_failed} test(s) failed. Please review and fix.")
    exit(1)
