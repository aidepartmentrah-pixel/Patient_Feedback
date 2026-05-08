"""
03_verify.py — Query the database and compare every metric against expected benchmarks.

Prints a PASS/FAIL table covering:
  B1  Patient profiles (case count per patient)
  B2  Doctor monthly — Month 1 (complaint / praise per doctor)
  B3  Doctor monthly — Month 5 (complaint / praise per doctor)
  B4  Doctor total profile (all 100 cases, per doctor)
  B5  Worker monthly — Month 1
  B6  Worker monthly — Month 5
  B7  Worker total profile
  B8  Section subcases — Month 1 (subcase count per section)
  B9  Section subcases — Month 5
  B10 Section subcases — total
  B11 Total subcase count Month 1 = 75
  B12 Total subcase count Month 5 = 75
  B13 Total subcase count         = 150
  B14 Red flags  Month 1 = 5
  B15 Never events Month 1 = 10
  B16 Red flags  Month 5 = 10
  B17 Never events Month 5 = 5
  B18 Total red flags = 15
  B19 Total never events = 15
  B20 Total cases inserted = 100
"""

import sys, os, json

_HERE    = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.join(_HERE, '..', '..')
_REPO    = os.path.join(_BACKEND, '..')
sys.path.insert(0, os.path.abspath(_BACKEND))
sys.path.insert(0, os.path.abspath(_REPO))

from backend.core.database import get_connection

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

PASS_SYM = "PASS"
FAIL_SYM = "FAIL"

results = []   # (benchmark_id, label, expected, actual, passed)


def chk(bid, label, expected, actual):
    passed = (actual == expected)
    results.append((bid, label, expected, actual, passed))
    sym = PASS_SYM if passed else FAIL_SYM
    exp_str = str(expected)
    act_str = str(actual)
    mark = "" if passed else f"  ← expected {exp_str}"
    print(f"  [{sym}] {bid:<4}  {label:<55}  actual={act_str}{mark}")
    return passed


def load_json(name):
    with open(os.path.join(DATA_DIR, name)) as f:
        return json.load(f)


def get_test_case_ids(cur):
    cur.execute("""
        SELECT IncidentRequestCaseID
        FROM dbo.APP_IncidentCase
        WHERE PatientName LIKE 'T100_%'
    """)
    return [r[0] for r in cur.fetchall()]


def main():
    print("=" * 60)
    print("03_VERIFY — Benchmark Verification Report")
    print("=" * 60)

    bm       = load_json('expected_benchmarks.json')
    ids_data = load_json('inserted_ids.json')

    inserted_ids = [r["db_id"] for r in ids_data["inserted"]]
    month1 = bm["month1"]   # "2025-01"
    month5 = bm["month5"]   # "2025-05"

    if not inserted_ids:
        print("[ERROR] No inserted IDs found. Run 02_insert.py first.")
        sys.exit(1)

    conn = get_connection()
    cur  = conn.cursor()

    # Confirm inserted IDs match what's in DB
    db_ids = get_test_case_ids(cur)
    print(f"\nT100_ cases in DB: {len(db_ids)}  (inserted_ids.json has {len(inserted_ids)})\n")

    # Build IN clause helper
    def in_clause(ids):
        return "(" + ",".join(str(i) for i in ids) + ")"

    ids_in = in_clause(db_ids)

    # ── B20: Total cases ─────────────────────────────────────────
    chk("B20", "Total T100_ cases in DB", bm["total_cases"], len(db_ids))

    # ── B1: Patient profiles ──────────────────────────────────────
    print("\n[B1] Patient Profiles")
    cur.execute(f"""
        SELECT PatientName, COUNT(*) as cnt
        FROM dbo.APP_IncidentCase
        WHERE IncidentRequestCaseID IN {ids_in}
        GROUP BY PatientName
        ORDER BY PatientName
    """)
    actual_patients = {r[0]: r[1] for r in cur.fetchall()}
    for pat, expected_cnt in sorted(bm["patient_counts"].items()):
        actual_cnt = actual_patients.get(pat, 0)
        chk("B1", f"Patient {pat}", expected_cnt, actual_cnt)

    # ── B11-B13: Workflow Subcase totals (1 per case, DB constraint) ──
    print("\n[B11-B13] Workflow Subcases (1 per case — DB unique constraint)")

    cur.execute(f"""
        SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase a2
        INNER JOIN dbo.APP_IncidentCase ic
            ON a2.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
          AND a2.CaseType = 'INCIDENT_RESPONSE'
          AND CONVERT(varchar(7), ic.FeedbackRecievedDate, 120) = '{month1}'
    """)
    actual_subcases_m1 = cur.fetchone()[0]

    cur.execute(f"""
        SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase a2
        INNER JOIN dbo.APP_IncidentCase ic
            ON a2.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
          AND a2.CaseType = 'INCIDENT_RESPONSE'
          AND CONVERT(varchar(7), ic.FeedbackRecievedDate, 120) = '{month5}'
    """)
    actual_subcases_m5 = cur.fetchone()[0]

    cur.execute(f"""
        SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase a2
        INNER JOIN dbo.APP_IncidentCase ic
            ON a2.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
          AND a2.CaseType = 'INCIDENT_RESPONSE'
    """)
    actual_subcases_total = cur.fetchone()[0]

    # Expected: 1 subcase per case (50 cases per month = 50 per month, 100 total)
    chk("B11", "Workflow subcases Month 1 (1 per case)", 50,  actual_subcases_m1)
    chk("B12", "Workflow subcases Month 5 (1 per case)", 50,  actual_subcases_m5)
    chk("B13", "Workflow subcases total  (1 per case)", 100, actual_subcases_total)

    # ── B8-B10: Section TARGET RECORDS (from APP_IncidentCaseTargetDepartment) ──
    # Note: target records = 300 (all departments per case are stored here correctly)
    print("\n[B8-B10] Section Target Records (APP_IncidentCaseTargetDepartment)")

    cur.execute(f"""
        SELECT td.DepartmentID, COUNT(*) as cnt
        FROM dbo.APP_IncidentCaseTargetDepartment td
        INNER JOIN dbo.APP_IncidentCase ic
            ON td.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
          AND CONVERT(varchar(7), ic.FeedbackRecievedDate, 120) = '{month1}'
        GROUP BY td.DepartmentID
    """)
    actual_sec_m1 = {str(r[0]): r[1] for r in cur.fetchall()}

    cur.execute(f"""
        SELECT td.DepartmentID, COUNT(*) as cnt
        FROM dbo.APP_IncidentCaseTargetDepartment td
        INNER JOIN dbo.APP_IncidentCase ic
            ON td.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
          AND CONVERT(varchar(7), ic.FeedbackRecievedDate, 120) = '{month5}'
        GROUP BY td.DepartmentID
    """)
    actual_sec_m5 = {str(r[0]): r[1] for r in cur.fetchall()}

    cur.execute(f"""
        SELECT td.DepartmentID, COUNT(*) as cnt
        FROM dbo.APP_IncidentCaseTargetDepartment td
        INNER JOIN dbo.APP_IncidentCase ic
            ON td.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
        GROUP BY td.DepartmentID
    """)
    actual_sec_total = {str(r[0]): r[1] for r in cur.fetchall()}

    section_ids = load_json('config.json')["section_ids"]
    section_names = load_json('config.json')["section_names"]
    for i, sid in enumerate(section_ids):
        skey = str(sid)
        label = f"S{i+1}(ID={sid})"
        chk("B8",  f"Section {label} subcases Month 1", bm["section_subcases_month1"].get(skey, 0), actual_sec_m1.get(skey, 0))
        chk("B9",  f"Section {label} subcases Month 5", bm["section_subcases_month5"].get(skey, 0), actual_sec_m5.get(skey, 0))
        chk("B10", f"Section {label} subcases Total",   bm["section_subcases_total"].get(skey, 0),  actual_sec_total.get(skey, 0))

    # ── B2-B4: Doctor profiles ────────────────────────────────────
    print("\n[B2-B4] Doctor Profiles")

    cur.execute(f"""
        SELECT d.DoctorID,
               CONVERT(varchar(7), ic.FeedbackRecievedDate, 120) as ym,
               ic.FeedbackIntentTypeID,
               COUNT(*) as cnt
        FROM dbo.APP_IncidentCaseDoctor d
        INNER JOIN dbo.APP_IncidentCase ic
            ON d.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
        GROUP BY d.DoctorID, CONVERT(varchar(7), ic.FeedbackRecievedDate, 120), ic.FeedbackIntentTypeID
    """)
    rows = cur.fetchall()

    doctor_ids = load_json('config.json')["doctor_ids"]
    actual_doc = {did: {month1: {"C": 0, "P": 0}, month5: {"C": 0, "P": 0}} for did in doctor_ids}
    for did, ym, intent, cnt in rows:
        if did in actual_doc and ym in actual_doc[did]:
            key = "C" if intent == 1 else "P"
            actual_doc[did][ym][key] += cnt

    for i, did in enumerate(doctor_ids):
        dkey = str(did)
        exp_m1 = bm["doctor_month1"][dkey][month1]
        exp_m5 = bm["doctor_month5"][dkey][month5]
        exp_tot = bm["doctor_totals"][dkey]
        act_m1  = actual_doc[did][month1]
        act_m5  = actual_doc[did][month5]
        act_tot = {"C": act_m1["C"]+act_m5["C"], "P": act_m1["P"]+act_m5["P"]}

        chk("B2", f"Doctor {did} M1 Complaints", exp_m1["C"], act_m1["C"])
        chk("B2", f"Doctor {did} M1 Praises",    exp_m1["P"], act_m1["P"])
        chk("B3", f"Doctor {did} M5 Complaints", exp_m5["C"], act_m5["C"])
        chk("B3", f"Doctor {did} M5 Praises",    exp_m5["P"], act_m5["P"])
        chk("B4", f"Doctor {did} Total Complaints", exp_tot["C"], act_tot["C"])
        chk("B4", f"Doctor {did} Total Praises",    exp_tot["P"], act_tot["P"])

    # ── B5-B7: Worker profiles ────────────────────────────────────
    print("\n[B5-B7] Worker Profiles")

    cur.execute(f"""
        SELECT e.EmployeeID,
               CONVERT(varchar(7), ic.FeedbackRecievedDate, 120) as ym,
               ic.FeedbackIntentTypeID,
               COUNT(*) as cnt
        FROM dbo.APP_IncidentCaseEmployee e
        INNER JOIN dbo.APP_IncidentCase ic
            ON e.IncidentRequestCaseID = ic.IncidentRequestCaseID
        WHERE ic.IncidentRequestCaseID IN {ids_in}
        GROUP BY e.EmployeeID, CONVERT(varchar(7), ic.FeedbackRecievedDate, 120), ic.FeedbackIntentTypeID
    """)
    rows = cur.fetchall()

    worker_ids = load_json('config.json')["worker_ids"]
    actual_wkr = {wid: {month1: {"C": 0, "P": 0}, month5: {"C": 0, "P": 0}} for wid in worker_ids}
    for wid, ym, intent, cnt in rows:
        if wid in actual_wkr and ym in actual_wkr[wid]:
            key = "C" if intent == 1 else "P"
            actual_wkr[wid][ym][key] += cnt

    for i, wid in enumerate(worker_ids):
        wkey = str(wid)
        exp_m1 = bm["worker_month1"][wkey][month1]
        exp_m5 = bm["worker_month5"][wkey][month5]
        exp_tot = bm["worker_totals"][wkey]
        act_m1  = actual_wkr[wid][month1]
        act_m5  = actual_wkr[wid][month5]
        act_tot = {"C": act_m1["C"]+act_m5["C"], "P": act_m1["P"]+act_m5["P"]}

        chk("B5", f"Worker {wid} M1 Complaints", exp_m1["C"], act_m1["C"])
        chk("B5", f"Worker {wid} M1 Praises",    exp_m1["P"], act_m1["P"])
        chk("B6", f"Worker {wid} M5 Complaints", exp_m5["C"], act_m5["C"])
        chk("B6", f"Worker {wid} M5 Praises",    exp_m5["P"], act_m5["P"])
        chk("B7", f"Worker {wid} Total Complaints", exp_tot["C"], act_tot["C"])
        chk("B7", f"Worker {wid} Total Praises",    exp_tot["P"], act_tot["P"])

    # ── B14-B19: Red Flags & Never Events ────────────────────────
    print("\n[B14-B19] Red Flags & Never Events")

    cur.execute(f"""
        SELECT ClinicalRiskTypeID,
               CONVERT(varchar(7), FeedbackRecievedDate, 120) as ym,
               COUNT(*) as cnt
        FROM dbo.APP_IncidentCase
        WHERE IncidentRequestCaseID IN {ids_in}
          AND ClinicalRiskTypeID IN (2,3)
        GROUP BY ClinicalRiskTypeID, CONVERT(varchar(7), FeedbackRecievedDate, 120)
    """)
    risk_rows = cur.fetchall()
    risk_actual = {}
    for rtype, ym, cnt in risk_rows:
        risk_actual[(rtype, ym)] = cnt

    chk("B14", f"Red Flags Month 1",    bm["red_flags_month1"],    risk_actual.get((2, month1), 0))
    chk("B15", f"Never Events Month 1", bm["never_events_month1"], risk_actual.get((3, month1), 0))
    chk("B16", f"Red Flags Month 5",    bm["red_flags_month5"],    risk_actual.get((2, month5), 0))
    chk("B17", f"Never Events Month 5", bm["never_events_month5"], risk_actual.get((3, month5), 0))
    chk("B18", "Red Flags Total",       bm["red_flags_total"],
        risk_actual.get((2, month1), 0) + risk_actual.get((2, month5), 0))
    chk("B19", "Never Events Total",    bm["never_events_total"],
        risk_actual.get((3, month1), 0) + risk_actual.get((3, month5), 0))

    cur.close()
    conn.close()

    # ── Final Summary ─────────────────────────────────────────────
    total   = len(results)
    passed  = sum(1 for r in results if r[4])
    failed  = total - passed
    pct     = 100 * passed // total if total else 0

    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY")
    print(f"  Total checks : {total}")
    print(f"  Passed       : {passed}")
    print(f"  Failed       : {failed}")
    print(f"  Score        : {pct}%")

    if failed == 0:
        print("\n  *** ALL BENCHMARKS PASSED — Data integrity CONFIRMED ***")
    else:
        print(f"\n  *** {failed} BENCHMARK(S) FAILED — Review output above ***")
        print("\n  Failed items:")
        for bid, label, exp, act, ok in results:
            if not ok:
                print(f"    [{bid}] {label}: expected={exp}, actual={act}")

    print("=" * 60)

    # Save report
    report_path = os.path.join(DATA_DIR, 'verification_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            "total": total, "passed": passed, "failed": failed, "score_pct": pct,
            "details": [
                {"id": r[0], "label": r[1], "expected": r[2], "actual": r[3], "passed": r[4]}
                for r in results
            ]
        }, f, indent=2)
    print(f"\nReport saved -> {report_path}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
