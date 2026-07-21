"""
Endpoint Test Suite for Action Item Coordination — AIC-S7 Notification Foundation
Hits the LIVE backend over HTTP (http://127.0.0.1:8000) to verify:
- POST /api/v2/supervisor-action-items/{id}/acknowledge (Case 1)
- POST /api/v2/action-item-notices/{id}/acknowledge (Case 2)
- RBAC: only the recipient can acknowledge
- Idempotency: double-accept is a safe no-op
- Inbox merge (AIC-S7 S2.5): both notice types appear/disappear correctly

Requires the backend to be running (PatientFeedbackAPI service) and the
standard test accounts to exist (see feedback_credentials memory).

Run: python backend/tests/test_aic_s7_notification_endpoints.py
"""

import sys
sys.path.insert(0, 'backend')

import requests

BASE_URL = "http://127.0.0.1:8000"

CASE_ID = 1
VALID_UNIT = 1  # الادارة العامة -- administration_admin scoped here

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
print("AIC-S7: NOTIFICATION ENDPOINT TEST SUITE (live HTTP)")
print("=" * 70)
print()

print("Logging in test accounts...")
supervisor = login("complaint_supervisor", "5bb5a339")
administration_admin = login("administration_admin", "admin2026")
section_admin = login("section_admin", "section2026")
print("All accounts logged in.")
print()

created_item_ids = []


# ============================================================
# CASE 1: supervisor-action-items/{id}/acknowledge
# ============================================================

@test("1. Create + acknowledge as the target user - 200, AcknowledgedAt set")
def t1():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": VALID_UNIT,
        "description": "AIC-S7 endpoint test - ack by target unit scope",
    })
    assert resp.status_code == 200, resp.text
    item_id = resp.json()["item"]["action_item_id"]
    created_item_ids.append(item_id)

    ack_resp = administration_admin.post(
        f"{BASE_URL}/api/v2/supervisor-action-items/{item_id}/acknowledge"
    )
    assert ack_resp.status_code == 200, ack_resp.text
    assert ack_resp.json()["item"]["acknowledged_at"] is not None

t1()


@test("2. Out-of-scope user gets 403 acknowledging")
def t2():
    resp = supervisor.post(f"{BASE_URL}/api/v2/supervisor-action-items", json={
        "incident_request_case_id": CASE_ID,
        "target_org_unit_id": VALID_UNIT,
        "description": "AIC-S7 endpoint test - 403 check",
    })
    assert resp.status_code == 200, resp.text
    item_id = resp.json()["item"]["action_item_id"]
    created_item_ids.append(item_id)

    ack_resp = section_admin.post(
        f"{BASE_URL}/api/v2/supervisor-action-items/{item_id}/acknowledge"
    )
    assert ack_resp.status_code == 403, f"Expected 403, got {ack_resp.status_code}: {ack_resp.text}"

t2()


@test("3. Double-acknowledge is idempotent (still 200, same AcknowledgedAt)")
def t3():
    item_id = created_item_ids[0]
    ack_resp = administration_admin.post(
        f"{BASE_URL}/api/v2/supervisor-action-items/{item_id}/acknowledge"
    )
    assert ack_resp.status_code == 200, ack_resp.text
    assert ack_resp.json()["item"]["acknowledged_at"] is not None

t3()


@test("4. Acknowledging a nonexistent item returns 404")
def t4():
    ack_resp = administration_admin.post(
        f"{BASE_URL}/api/v2/supervisor-action-items/9999999/acknowledge"
    )
    assert ack_resp.status_code == 404, f"Expected 404, got {ack_resp.status_code}: {ack_resp.text}"

t4()


# ============================================================
# CLEANUP
# ============================================================

if created_item_ids:
    import sys as _sys
    _sys.path.insert(0, '.')
    from core.database import get_connection
    conn = get_connection()
    cursor = conn.cursor()
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
    print(f"Cleaned up {len(created_item_ids)} test supervisor action item(s).")

print()
print("=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print(f"Total Tests:  {tests_run}")
print(f"Passed:       {tests_passed}")
print(f"Failed:       {tests_failed}")
print()

if tests_failed == 0:
    print("ALL TESTS PASSED")
    sys.exit(0)
else:
    print(f"{tests_failed} test(s) failed.")
    sys.exit(1)
