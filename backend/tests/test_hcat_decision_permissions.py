"""
HCAT (Patient Services) decision: everyone involved can VIEW it, only HCAT
can WRITE it, and it closes once every level that exists has clicked OK.

Real HTTP against the running backend (default http://localhost:8100), real
logins. Test users' passwords come from the credentials export store, so
the dev/test DB must already have them (Users Management > Import
Passwords); nothing is printed.

Fixtures (all deleted at the end):
  A: case -> OPD-style section directly under an Administration (no Department)
  B: case -> section under a Department (all three levels)
  C/D: stuck "decision completed" cases for migration 004

Checks:
  1. section / administration admins get 403 saving the decision; WORKER too
  2. HCAT (COMPLAINT_SUPERVISOR) saves -> 200, status COMPLETED
  3. section admin SEES the decision (inbox item with the text, acknowledge only)
  4. after OK: section archive item is view-only but still carries the text;
     a save attempt is still 403 and the text is unchanged
  5. A (no Department): section + administration OK -> DECISION_ACKNOWLEDGED
  6. B: section + administration OK -> still COMPLETED; + department -> ACKNOWLEDGED
  7. HCAT can still re-edit a completed decision (200)
  8. migration 004 finalizes stuck C (all existing levels acked), leaves D alone
     NOTE: step 8 runs the real migration against the whole database, so it
     also finalizes any genuinely stuck cases there -- which is its purpose.

Run: python backend/tests/test_hcat_decision_permissions.py [base_url]
"""

import os
import re
import sys
import traceback

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(backend_dir))
sys.path.insert(0, backend_dir)

import requests

from backend.core.database import get_connection
from backend.api.schemas.auth_models import CurrentUser, UserScope
from backend.api.services.user_credentials_service import get_all_user_credentials_service
from backend.api_v2.services import case_response_service
from backend.tests.test_inspect_imported_case import create_imported_case, cleanup, fetch_one

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100"
MIGRATION_004 = os.environ.get("MIGRATION_004") or os.path.join(
    os.path.dirname(backend_dir), "database", "sqlserver", "migrations", "004_finalize_acknowledged_decisions.sql")
DECISION = "TEST HCAT decision text"

results = []


def check(name, ok, extra=""):
    results.append((name, bool(ok)))
    print(f"{'PASS' if ok else 'FAIL'} {name}{' -- ' + extra if extra and not ok else ''}")


def execute(sql, params=()):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def subcase(subcase_id):
    return fetch_one("SELECT Status, PatientServicesDecisionText FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = ?",
                     (subcase_id,))


def create_subcase(case_id, unit_id, status, decision=None):
    supervisor_id = fetch_one("SELECT UserID FROM dbo.APP_Users WHERE Username = 'complaint_supervisor'").UserID
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO dbo.APP_AdministrativeSubcase (CaseType, IncidentRequestCaseID, TargetOrgUnitID, Status, "
            "CreatedByUserID, PatientServicesDecisionText) OUTPUT INSERTED.SubcaseID VALUES ('INCIDENT_RESPONSE', ?, ?, ?, ?, ?)",
            (case_id, unit_id, status, supervisor_id, decision))
        sid = cur.fetchone()[0]
        conn.commit()
        return sid
    finally:
        conn.close()


def add_ack(subcase_id, level):
    execute("INSERT INTO dbo.APP_SubcaseDecisionAcknowledgment (SubcaseID, OrgLevel, AcknowledgedByUserID, AcknowledgedAt) "
            "VALUES (?, ?, (SELECT TOP 1 UserID FROM dbo.APP_Users), GETDATE())", (subcase_id, level))


_passwords = None


def session(username):
    global _passwords
    if _passwords is None:
        _passwords = {c["username"]: c["test_password"] for c in get_all_user_credentials_service()}
    assert _passwords.get(username), f"no stored password for {username} -- import passwords first"
    s = requests.Session()
    r = s.post(f"{BASE}/api/auth/login", json={"username": username, "password": _passwords[username]}, timeout=30)
    assert r.status_code == 200, f"login {username}: {r.status_code}"
    return s


def act(s, subcase_id, action, payload=None):
    return s.post(f"{BASE}/api/v2/workflow/case/{subcase_id}/act",
                  json={"action": action, "payload": payload or {}}, timeout=30)


def find_item(items, subcase_id):
    if isinstance(items, dict):
        items = items.get("items") or items.get("inbox") or items.get("archive") or []
    return next((i for i in items if i.get("subcase_id") == subcase_id), None)


def run():
    made = []  # (incident_id, case_id, subcase_ids)
    try:
        def fixture(unit_id):
            created = {}
            incident_id, case_id, _ = create_imported_case(created, target_unit_id=unit_id)
            made.append((incident_id, case_id))
            return case_id

        # A: OPD (section 71 directly under Administration 4)
        a = create_subcase(fixture(71), 71, "WAITING_PATIENT_SERVICES_DECISION")
        # B: section 31 -> Department 5 -> Administration 1
        b = create_subcase(fixture(31), 31, "WAITING_PATIENT_SERVICES_DECISION")

        sup, sec71, adm4 = session("complaint_supervisor"), session("sec_71_admin"), session("adm_4_admin")

        # 1. non-HCAT users cannot write
        for who, s in (("section admin", sec71), ("administration admin", adm4)):
            r = act(s, a, "SAVE_PATIENT_SERVICES_DECISION", {"decision_text": "hijack"})
            check(f"1 {who} saving the HCAT decision -> 403", r.status_code == 403, f"HTTP {r.status_code}")
        worker = CurrentUser(user_id=1, username="worker", is_active=True, roles=["WORKER"],
                             scopes=[UserScope(role_code="WORKER", org_unit_id=1, org_unit_type="ADMINISTRATION")])
        try:
            case_response_service.save_patient_services_decision(a, "hijack", worker)
            check("1 WORKER saving the HCAT decision -> refused", False, "no error raised")
        except Exception as e:
            check("1 WORKER saving the HCAT decision -> refused", getattr(e, "status_code", None) == 403, repr(e))
        check("1 nothing was written", subcase(a).PatientServicesDecisionText is None)

        # 2. HCAT writes
        r = act(sup, a, "SAVE_PATIENT_SERVICES_DECISION", {"decision_text": DECISION})
        check("2 HCAT saves the decision -> 200", r.status_code == 200, r.text[:150])
        check("2 status is PATIENT_SERVICES_DECISION_COMPLETED",
              subcase(a).Status == "PATIENT_SERVICES_DECISION_COMPLETED")

        # 3. section admin sees it, can only acknowledge
        item = find_item(sec71.get(f"{BASE}/api/v2/workflow/inbox", timeout=30).json(), a)
        check("3 section admin inbox shows the decision", item is not None)
        if item:
            check("3 inbox item carries the decision text", item.get("patient_services_decision_text") == DECISION)
            check("3 inbox item offers acknowledge, not edit",
                  "acknowledge_decision" in item.get("allowed_actions", [])
                  and "edit_patient_services_decision" not in item.get("allowed_actions", []),
                  str(item.get("allowed_actions")))

        # 4. OK -> archive is view-only but still shows the decision
        r = act(sec71, a, "ACKNOWLEDGE_DECISION")
        check("4 section admin clicks OK -> 200", r.status_code == 200, r.text[:150])
        arch = find_item(sec71.get(f"{BASE}/api/v2/workflow/inbox/archive", timeout=30).json(), a)
        check("4 case is in the section archive", arch is not None)
        if arch:
            check("4 archive item is view-only", arch.get("allowed_actions") == ["view"], str(arch.get("allowed_actions")))
            check("4 archive item still shows the decision text", arch.get("patient_services_decision_text") == DECISION)
        r = act(sec71, a, "SAVE_PATIENT_SERVICES_DECISION", {"decision_text": "hijack after archive"})
        check("4 section admin save from archive -> 403", r.status_code == 403, f"HTTP {r.status_code}")
        check("4 decision text unchanged", subcase(a).PatientServicesDecisionText == DECISION)

        # 5. no Department: section + administration OK closes it
        r = act(adm4, a, "ACKNOWLEDGE_DECISION")
        check("5 administration admin clicks OK -> 200", r.status_code == 200, r.text[:150])
        check("5 no-Department case -> DECISION_ACKNOWLEDGED", subcase(a).Status == "DECISION_ACKNOWLEDGED",
              subcase(a).Status)

        # 6. three-level case needs all three
        act(sup, b, "SAVE_PATIENT_SERVICES_DECISION", {"decision_text": DECISION})
        for user in ("sec_31_admin", "adm_1_admin"):
            r = act(session(user), b, "ACKNOWLEDGE_DECISION")
            check(f"6 {user} clicks OK -> 200", r.status_code == 200, r.text[:150])
        check("6 without Department OK -> still COMPLETED", subcase(b).Status == "PATIENT_SERVICES_DECISION_COMPLETED",
              subcase(b).Status)
        r = act(session("dept_5_admin"), b, "ACKNOWLEDGE_DECISION")
        check("6 dept_5_admin clicks OK -> 200", r.status_code == 200, r.text[:150])
        check("6 all three OK -> DECISION_ACKNOWLEDGED", subcase(b).Status == "DECISION_ACKNOWLEDGED", subcase(b).Status)

        # 7. HCAT can still re-edit a completed decision
        c7 = create_subcase(fixture(71), 71, "PATIENT_SERVICES_DECISION_COMPLETED", DECISION)
        r = act(sup, c7, "SAVE_PATIENT_SERVICES_DECISION", {"decision_text": DECISION + " (edited)"})
        check("7 HCAT re-edits a completed decision -> 200", r.status_code == 200, r.text[:150])
        check("7 edit stored", subcase(c7).PatientServicesDecisionText == DECISION + " (edited)")

        # 8. migration 004: finalize stuck cases only when every existing level acked
        c = create_subcase(fixture(71), 71, "PATIENT_SERVICES_DECISION_COMPLETED", DECISION)
        add_ack(c, "section"); add_ack(c, "administration")
        d = create_subcase(fixture(31), 31, "PATIENT_SERVICES_DECISION_COMPLETED", DECISION)
        add_ack(d, "section"); add_ack(d, "administration")
        sql = open(MIGRATION_004, encoding="utf-8").read()
        for batch in [x for x in re.split(r"^\s*GO\s*$", sql, flags=re.M) if x.strip()]:
            if "SchemaMigrationHistory" not in batch:  # don't record the test run
                execute(batch)
        check("8 migration 004 finalizes stuck no-Department case", subcase(c).Status == "DECISION_ACKNOWLEDGED",
              subcase(c).Status)
        check("8 migration 004 leaves case missing its Department OK", subcase(d).Status == "PATIENT_SERVICES_DECISION_COMPLETED",
              subcase(d).Status)
    except Exception as e:
        traceback.print_exc()
        check("unexpected error", False, repr(e))
    finally:
        for incident_id, case_id in made:
            execute("DELETE FROM dbo.APP_SubcaseDecisionAcknowledgment WHERE SubcaseID IN "
                    "(SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID = ?)", (case_id,))
            cleanup(incident_id, case_id)  # also deletes the subcases
        print(f"cleaned up {len(made)} test case(s)")

    failed = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
