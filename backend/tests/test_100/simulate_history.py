"""
simulate_history.py -- Fill the dev DB with ~1200 realistic cases (2025-01-01 -> today)
so dashboards/reports/workflow screens can be exercised like the offline deployment.

How it works (fidelity over speed):
  1. Each case is created through the real create_record() path (validation, ML job
     registration, subcase creation).
  2. The subcase is then driven through the REAL workflow functions
     (approve_department / reject_department / approve_administration /
     save_patient_services_decision / acknowledge_patient_services_decision /
     force_close_subcase) using a stand-in user, so statuses, action-item
     transitions and decision acknowledgments are exactly what the app writes.
  3. All timestamps the app stamped "now" are shifted back to the simulated date,
     with simulated response delays applied, so aging/deadline/late-reply
     reports see a believable history.
  4. Finally the automatic force-close scan runs once (if the policy is enabled),
     exactly as it would have over the deployment's lifetime.

Patient names carry the T100_SIM_ prefix. This script has its own --cleanup that
walks the foreign keys (99_cleanup.py predates the ml.* tables and fails on them).

Run inside the dev backend container:
  docker exec -e PYTHONPATH=/app:/app/backend -w /app/backend pfms-dev-backend \
      python tests/test_100/simulate_history.py [--count 1200] [--seed 20261013]
  ... python tests/test_100/simulate_history.py --cleanup
"""

import sys, os, random, datetime, argparse, collections
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

from backend.core.database import get_connection
from backend.api.services.insert_service import create_record

PREFIX = "T100_SIM_"
START = datetime.date(2025, 1, 1)
TYPE_DEPT, TYPE_SECTION = 324, 325
ACTION_TITLES = [
    "Retrain staff on the escalation procedure", "Review and update the unit SOP",
    "Hold a counselling session with the involved staff", "Add signage / patient information leaflet",
    "Audit the next 20 cases for recurrence", "Fix equipment / environment issue reported",
    "Brief the shift team on the incident", "Follow up with the patient's family",
]
EXPLANATIONS = [
    "Reviewed with the team; the issue was confirmed and corrective steps were agreed.",
    "The reported event was investigated and the root cause identified.",
    "Staff were counselled and the unit procedure was reinforced.",
]


# --------------------------------------------------------------------------- helpers
def stand_in_user(role, unit_id=None, uid=1):
    scope = SimpleNamespace(role_code=role, unit_id=unit_id)
    return SimpleNamespace(
        user_id=uid, roles=[role], scopes=[scope],
        allowed_unit_ids={unit_id} if unit_id else set(),
    )


def q(cur, sql, *args):
    cur.execute(sql, *args)
    return [tuple(r) for r in cur.fetchall()]


def zipf_weights(n, s=1.1):
    return [1.0 / ((i + 1) ** s) for i in range(n)]


def build_dates(count, today):
    """Weekday-heavy, gently growing volume, ascending so IDs follow time."""
    days = [START + datetime.timedelta(days=i) for i in range((today - START).days + 1)]
    span = max(len(days) - 1, 1)
    weights = [(0.35 if d.weekday() >= 4 else 1.0) * (0.8 + 0.5 * i / span) for i, d in enumerate(days)]
    return sorted(random.choices(days, weights=weights, k=count))


OUTCOMES = {  # by age bucket: weights over (pending_dept, pending_admin, returned, approved, decided, acked, force)
    7:   (70, 20, 5, 5, 0, 0, 0),
    30:  (20, 20, 7, 20, 13, 15, 5),
    120: (6, 6, 5, 12, 15, 50, 6),
    9999:(3, 3, 3, 6, 10, 70, 5),
}
ORDER = ["pending_dept", "pending_admin", "returned", "approved", "decided", "acked", "force"]


def pick_outcome(age):
    for limit, w in OUTCOMES.items():
        if age < limit:
            return random.choices(ORDER, weights=w)[0]


# --------------------------------------------------------------------------- seeding
def load_reference(cur):
    taxonomy = q(cur, """
        SELECT cl.ClassificationID, sc.SubCategoryID, ca.CategoryID, ca.DomainID
        FROM dbo.APP_LOOKUP_CLASSIFICATION cl
        JOIN dbo.APP_LOOKUP_SUBCATEGORY sc ON sc.SubCategoryID = cl.SubCategoryID
        JOIN dbo.APP_LOOKUP_CATEGORY ca ON ca.CategoryID = sc.CategoryID
        WHERE cl.IsActive = 1""")
    sections = [r[0] for r in q(cur, "SELECT UniqueID FROM dbo.AdminsrationUnit WHERE Type=? AND ISNULL(Frozen,0)=0", TYPE_SECTION)]
    depts = [r[0] for r in q(cur, "SELECT UniqueID FROM dbo.AdminsrationUnit WHERE Type=? AND ISNULL(Frozen,0)=0", TYPE_DEPT)]
    deadlines = dict(q(cur, "SELECT SettingKey, SettingValue FROM dbo.APP_SystemSettings WHERE SettingKey LIKE '%deadline_days'"))
    return taxonomy, sections, depts, deadlines


def build_payload(i, received, tax, section, issuer, is_notice):
    cls, sub, cat, dom = tax
    risk = random.choices([1, 2, 3], weights=[90, 7, 3])[0]
    return {
        "complaint_text": f"SIM history case {i} - {received.isoformat()}",
        "feedback_received_date": received.isoformat(),
        "incident_date": (received - datetime.timedelta(days=random.randint(0, 4))).isoformat(),
        "issuing_department_id": issuer,
        "domain_id": dom, "category_id": cat, "subcategory_id": sub, "classification_id": cls,
        "severity_id": random.choices([1, 2, 3], weights=[50, 35, 15])[0],
        "stage_id": random.randint(1, 6), "harm_id": random.randint(1, 5),
        "clinical_risk_type_id": risk,
        "feedback_intent_type_id": 2 if is_notice else 1,
        "requires_explanation": not is_notice,
        "immediate_action": "Immediate action recorded", "taken_action": "Action taken recorded",
        "patient_name": f"{PREFIX}{i:04d}",
        "is_inpatient": random.random() < 0.6,
        "source_id": random.choices(range(1, 9), weights=[10, 8, 25, 12, 8, 12, 10, 15])[0],
        "building_id": 1,
        "target_department_ids": [section],
        "doctors": [{"doctor_id": 101 + (i % 25), "doctor_name": f"Dr. Test Doctor {i % 25 + 1}"}] if random.random() < 0.6 else [],
        "employees": [{"employee_id": 6 + (i % 20), "full_name": f"Worker Number {i % 20 + 1}"}] if random.random() < 0.5 else [],
    }


def add_action_items(cur, subcase_id, created):
    from backend.api_v2.db_layer import action_item_subcase_db as ai
    for title in random.sample(ACTION_TITLES, random.randint(1, 3)):
        ai.create_action_item(
            subcase_id=subcase_id, title=title, description=None,
            due_date=(created + datetime.timedelta(days=random.randint(10, 60))).date(),
            created_by_user_id=1, initial_status="SUBMITTED_TO_DEPT",
        )


def drive_workflow(cur, subcase_id, section, outcome, is_notice, delays):
    """Run the real service functions. Returns the final subcase status."""
    from backend.api_v2.services import case_response_service as crs
    sup = stand_in_user("COMPLAINT_SUPERVISOR", section)
    sup.roles.append("SOFTWARE_ADMIN")
    if outcome == "pending_dept":
        return
    if outcome == "returned":
        crs.reject_department(subcase_id, "Response incomplete - please revise and resubmit.", sup)
        return
    if outcome == "force":
        crs.force_close_subcase(subcase_id, "Closed administratively by the supervisor.", sup)
        return
    crs.approve_department(subcase_id, sup)
    if outcome == "pending_admin":
        return
    crs.approve_administration(subcase_id, sup)
    if outcome == "approved" or is_notice:
        return
    cur.execute("SELECT Status FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID=?", subcase_id)
    if cur.fetchone()[0] != "WAITING_PATIENT_SERVICES_DECISION":
        return  # non-complaint risk types (red flag / never event) end at ADMIN_APPROVED
    crs.save_patient_services_decision(subcase_id, "Decision recorded according to the scientific references.", sup)
    if outcome == "decided":
        return
    for role in ("SECTION_ADMIN", "DEPARTMENT_ADMIN", "ADMINISTRATION_ADMIN"):
        try:
            crs.acknowledge_patient_services_decision(subcase_id, stand_in_user(role, section))
        except Exception:
            break  # already finalized / level not applicable


def backdate(cur, case_id, subcase_id, created, delays, deadlines):
    """Shift app-stamped 'now' values to the simulated timeline."""
    now = datetime.datetime.now()
    shift = int((created - now).total_seconds())
    total = sum(delays.values())

    def shift_all(table, where_col, where_val):
        cols = [r[0] for r in q(cur, """
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA='dbo' AND TABLE_NAME=? AND DATA_TYPE IN ('datetime','datetime2','smalldatetime')""", table)]
        if cols:
            sets = ", ".join(f"[{c}] = DATEADD(second, ?, [{c}])" for c in cols)
            cur.execute(f"UPDATE dbo.{table} SET {sets} WHERE {where_col} = ?", *([shift] * len(cols)), where_val)

    shift_all("APP_IncidentCase", "IncidentRequestCaseID", case_id)
    shift_all("APP_AdministrativeSubcase", "SubcaseID", subcase_id)
    shift_all("APP_SubcaseActionItem", "SubcaseID", subcase_id)
    shift_all("APP_SubcaseDecisionAcknowledgment", "SubcaseID", subcase_id)

    # the admin deadline was seeded at (approve-time + N days); move it after the dept delay
    cur.execute("""UPDATE dbo.APP_AdministrativeSubcase
                   SET AdministrationDeadlineAt = DATEADD(day, ?, AdministrationDeadlineAt),
                       UpdatedAt = DATEADD(day, ?, ?)
                   WHERE SubcaseID=?""", delays.get("dept", 0), total, created, subcase_id)

    dept_days = int(deadlines.get("department_deadline_days", 14) or 14)
    admin_days = int(deadlines.get("administration_deadline_days", 14) or 14)
    if delays.get("dept", 0) > dept_days:
        cur.execute("UPDATE dbo.APP_AdministrativeSubcase SET DepartmentLateReply=1 WHERE SubcaseID=?", subcase_id)
    if delays.get("admin", 0) > admin_days:
        cur.execute("UPDATE dbo.APP_AdministrativeSubcase SET AdministrationLateReply=1 WHERE SubcaseID=?", subcase_id)


def seed(count):
    today = datetime.date.today()
    conn = get_connection(); conn.autocommit = False
    cur = conn.cursor()
    taxonomy, sections, depts, deadlines = load_reference(cur)
    print(f"{len(taxonomy)} classifications, {len(sections)} sections, {len(depts)} departments, deadlines={deadlines}")

    tax_w = zipf_weights(len(taxonomy), 0.6); random.shuffle(taxonomy)
    sec_w = zipf_weights(len(sections), 0.9); random.shuffle(sections)
    iss_w = zipf_weights(len(depts), 0.8); random.shuffle(depts)

    dates = build_dates(count, today)
    stats, failures = collections.Counter(), 0
    for i, d in enumerate(dates, 1):
        age = (today - d).days
        is_notice = random.random() < 0.10
        outcome = pick_outcome(age)
        section = random.choices(sections, weights=sec_w)[0]
        payload = build_payload(i, d, random.choices(taxonomy, weights=tax_w)[0], section,
                                random.choices(depts, weights=iss_w)[0], is_notice)
        res = create_record(payload)
        if not res.get("success"):
            failures += 1
            print(f"[FAIL] #{i} {d}: {res.get('message')}")
            continue
        case_id = int(res["id"])
        created = datetime.datetime.combine(d, datetime.time(random.randint(7, 15), random.randint(0, 59)))
        row = q(cur, "SELECT SubcaseID FROM dbo.APP_AdministrativeSubcase WHERE IncidentRequestCaseID=?", case_id)
        if not row:
            print(f"[WARN] #{i} case {case_id} has no subcase"); continue
        subcase_id = row[0][0]

        # simulated response delays (days) must fit inside the case's age
        dept = min(1 + int(random.expovariate(1 / 6)), max(age, 0))
        admin = min(1 + int(random.expovariate(1 / 5)), max(age - dept, 0))
        delays = {"dept": dept, "admin": admin}
        if outcome in ("approved", "decided", "acked") and dept + admin > age:
            outcome = "pending_admin"

        try:
            if not is_notice or outcome != "force":
                add_action_items(cur, subcase_id, created)
            drive_workflow(cur, subcase_id, section, outcome, is_notice, delays)
            backdate(cur, case_id, subcase_id, created, {"dept": dept, "admin": admin if outcome not in ("pending_dept", "pending_admin", "returned") else 0}, deadlines)
            conn.commit()
        except Exception as e:
            conn.rollback()
            failures += 1
            print(f"[FAIL] #{i} case {case_id} outcome={outcome}: {e}")
            continue
        stats[outcome] += 1
        if i % 100 == 0:
            print(f"  ...{i}/{count}")

    print("\nOutcome mix:", dict(stats), "| failures:", failures)
    try:
        from backend.api_v2.services import automatic_force_close_service as afc
        print("Automatic force-close scan:", afc.run_automatic_force_close_scan())
    except Exception as e:
        print("Automatic force-close scan skipped:", e)
    conn.close()


def seed_closed(unit_id, count, start_index=9000):
    """Case-level CLOSED cases (CaseStatusID=3, what the dashboard's closed_cases counts)
    for one target unit, created the way a bulk import lands them (save_mode='import_closed')."""
    conn = get_connection(); cur = conn.cursor()
    taxonomy, _, depts, _ = load_reference(cur)
    conn.close()
    today = datetime.date.today()
    for n in range(count):
        d = START + datetime.timedelta(days=random.randint(0, (today - START).days - 30))
        p = build_payload(start_index + n, d, random.choice(taxonomy), unit_id, random.choice(depts), False)
        p["patient_name"] = f"{PREFIX}CLOSED_{unit_id}_{n + 1:02d}"
        res = create_record(p, save_mode="import_closed")
        print(("ok  " if res.get("success") else "FAIL"), p["patient_name"], res.get("id") or res.get("message"))


# --------------------------------------------------------------------------- cleanup
def cleanup():
    """Delete every PREFIX case and everything that references it, discovered via FKs."""
    conn = get_connection(); cur = conn.cursor()
    ids = [r[0] for r in q(cur, "SELECT IncidentRequestCaseID FROM dbo.APP_IncidentCase WHERE PatientName LIKE ?", PREFIX + "%")]
    print(f"{len(ids)} {PREFIX}* cases found")
    if not ids:
        return
    fks = q(cur, """
        SELECT OBJECT_SCHEMA_NAME(f.parent_object_id), OBJECT_NAME(f.parent_object_id), COL_NAME(fc.parent_object_id, fc.parent_column_id),
               OBJECT_SCHEMA_NAME(f.referenced_object_id), OBJECT_NAME(f.referenced_object_id), COL_NAME(fc.referenced_object_id, fc.referenced_column_id)
        FROM sys.foreign_keys f JOIN sys.foreign_key_columns fc ON fc.constraint_object_id = f.object_id""")
    children = collections.defaultdict(list)
    for ps, pt, pc, rs, rt, rc in fks:
        if (ps, pt) != (rs, rt):
            children[(rs, rt)].append((ps, pt, pc, rc))

    cur.execute("CREATE TABLE #ids (id INT PRIMARY KEY)")
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        cur.execute("INSERT INTO #ids VALUES " + ",".join(f"({x})" for x in chunk))

    def delete(schema, table, where_sql):
        for cs, ct, ccol, rcol in children.get((schema, table), []):
            sub = f"SELECT [{rcol}] FROM [{schema}].[{table}] WHERE {where_sql}"
            delete(cs, ct, f"[{ccol}] IN ({sub})")
        cur.execute(f"DELETE FROM [{schema}].[{table}] WHERE {where_sql}")
        if cur.rowcount:
            print(f"  deleted {cur.rowcount:>5} from {schema}.{table}")

    try:
        delete("dbo", "APP_IncidentCase", "IncidentRequestCaseID IN (SELECT id FROM #ids)")
        conn.commit()
        print("Cleanup committed.")
    except Exception:
        conn.rollback(); raise
    finally:
        conn.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--closed-for", type=int, metavar="UNIT_ID", help="add --count case-level CLOSED cases for this unit")
    a = ap.parse_args()
    random.seed(a.seed)
    if a.cleanup: cleanup()
    elif a.closed_for: seed_closed(a.closed_for, a.count)
    else: seed(a.count)
