"""
dashboard_seed.py — Seed 100 cases each into cardiac 1/2/3 to exercise dashboard graphs.

Profiles (all via the real create_record() path, then optionally degraded):
  cardiac 1 (43, under Dept 28)  FULL      — every field populated, wide random spread
  cardiac 2 (44, under Dept 24)  GAPPY     — like a bulk import: some Stage/Harm/Category/Classification NULL
  cardiac 3 (45, under Dept 24)  NARROW    — one classification/stage, all High severity

Patient names use the T100_ prefix so 99_cleanup.py removes everything seeded here.

Run inside pfms-dev-backend:
  docker exec -e PYTHONPATH=/app -w /app/backend pfms-dev-backend python tests/test_100/dashboard_seed.py
"""

import sys, os, random, datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))

from backend.core.database import get_connection
from backend.api.services.insert_service import create_record

random.seed(20261002)
N_PER_SECTION = 100
ISSUERS = [95, 93, 60, 72, 98, 309, 83, 43, 44, 45]   # varied issuing units -> department chart
DOCTORS = [101, 102, 103, 104, 105]
WORKERS = [6, 7, 8, 9, 10]
START, END = datetime.date(2026, 4, 1), datetime.date(2026, 9, 28)


def load_taxonomy():
    conn = get_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT cl.ClassificationID, sc.SubCategoryID, ca.CategoryID, ca.DomainID
        FROM dbo.APP_LOOKUP_CLASSIFICATION cl
        JOIN dbo.APP_LOOKUP_SUBCATEGORY sc ON sc.SubCategoryID = cl.SubCategoryID
        JOIN dbo.APP_LOOKUP_CATEGORY ca ON ca.CategoryID = sc.CategoryID
        WHERE cl.IsActive = 1
    """)
    rows = [tuple(r) for r in cur.fetchall()]
    conn.close()
    return rows


def rand_date():
    return (START + datetime.timedelta(days=random.randint(0, (END - START).days))).isoformat()


def payload(target, idx, tag, tax, narrow=False):
    cls, sub, cat, dom = tax
    risk = random.choices([1, 2, 3], weights=[85, 10, 5])[0]
    intent = random.choices([1, 2], weights=[90, 10])[0]
    sev = 3 if narrow else random.choice([1, 2, 3])
    received = datetime.date.fromisoformat(rand_date())
    return {
        "complaint_text": f"DASHSEED {tag} #{idx}",
        "feedback_received_date": received.isoformat(),
        "incident_date": (received - datetime.timedelta(days=random.randint(0, 3))).isoformat(),
        "issuing_department_id": random.choice(ISSUERS),
        "domain_id": dom, "category_id": cat, "subcategory_id": sub, "classification_id": cls,
        "severity_id": sev,
        "stage_id": 1 if narrow else random.randint(1, 6),
        "harm_id": 3 if narrow else random.randint(1, 5),
        "clinical_risk_type_id": risk, "feedback_intent_type_id": intent,
        "requires_explanation": False,
        "immediate_action": "seed", "taken_action": "seed",
        "patient_name": f"T100_DASH_{tag}_{idx:03d}",
        "is_inpatient": True, "source_id": random.randint(1, 8), "building_id": 1,
        "target_department_ids": [target],
        "doctors": [{"doctor_id": DOCTORS[idx % 5], "doctor_name": f"Dr. Test Doctor {idx % 5 + 1}"}],
        "employees": [{"employee_id": WORKERS[idx % 5], "full_name": f"Worker Number {idx % 5 + 1}"}],
    }


def main():
    taxonomy = load_taxonomy()
    print(f"{len(taxonomy)} active classifications available")
    plan = [(43, "C1", False), (44, "C2", False), (45, "C3", True)]
    created = {}
    narrow_tax = random.choice(taxonomy)

    for target, tag, narrow in plan:
        ids = []
        for i in range(N_PER_SECTION):
            p = payload(target, i, tag, narrow_tax if narrow else random.choice(taxonomy), narrow)
            res = create_record(p)
            if not res.get("success"):
                print(f"[FAIL] {tag} #{i}: {res}")
                continue
            ids.append(int(res["id"]))
        created[tag] = ids
        print(f"{tag} (unit {target}): created {len(ids)}")

    # cardiac 2: degrade like a bulk import — NULL some fields
    ids = created["C2"]
    conn = get_connection(); cur = conn.cursor()
    degrade = [("StageID", 0.30), ("HarmLevelID", 0.30), ("ClassificationID", 0.25), ("CategoryID", 0.20)]
    for col, frac in degrade:
        pick = random.sample(ids, int(len(ids) * frac))
        for cid in pick:
            cur.execute(f"UPDATE dbo.APP_IncidentCase SET {col}=NULL WHERE IncidentRequestCaseID=?", cid)
        print(f"C2: NULLed {col} on {len(pick)} cases")
    conn.commit(); conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
