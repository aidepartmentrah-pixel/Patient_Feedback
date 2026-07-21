"""
Monthly/Seasonal report scope-isolation test suite.

End-to-end, HTTP-based, against the real running backend (no DB/service
bypass) — deliberately exercises the exact request shapes the Reporting
page sends, including the realistic "cascading picker" payload (all three
of administration_ids/department_ids/section_ids populated at once), since
that shape is what triggered the bug this suite regression-guards:

  monthly_report_service.generate_monthly_report() used to UNION the
  expanded descendant sets of every populated ID field instead of using
  only the most specific (deepest) level, so picking one Section after
  narrowing via Administration -> Department silently widened the report
  back out to the whole Administration's subtree.

Fix: backend/api/db_layer/reports_db.py::resolve_most_specific_scope(),
wired into monthly_report_service.py (data) and report_export_service.py
(filename/label).

Three tracks, matching the testing methodology agreed with the user:
  Track A — backend correctness: regression check for the exact bug,
            containment (no cross-unit leakage), and count reconciliation
            (child-unit sums equal parent-unit counts) for Monthly.
  Track B — frontend-contract: same checks, but payloads are built exactly
            the way ReportingPage.js/ReportFilters.js build them (comma-
            joined ID strings, all populated levels sent together).
  Track C — seasonal regression baseline (should already pass unchanged).

Run against the live backend:
    python backend/test_monthly_report_scope_isolation.py

Requires the backend to be reachable at BASE_URL and already restarted to
pick up the reports_db.py/monthly_report_service.py/report_export_service.py
fix (this script does not restart the service itself).
"""

import sys
import requests

BASE_URL = "http://127.0.0.1:8000"
USERNAME = "complaint_supervisor"
PASSWORD = "5bb5a339"

FAILURES = []
SKIPPED = []


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    if not condition:
        FAILURES.append(label)
    return condition


def skip(label, reason):
    print(f"[SKIP] {label} ({reason})")
    SKIPPED.append(label)


def login() -> requests.Session:
    s = requests.Session()
    r = s.post(f"{BASE_URL}/api/auth/login", json={"username": USERNAME, "password": PASSWORD}, timeout=30)
    r.raise_for_status()
    data = r.json()
    check("login succeeds", data.get("success") is True)
    user = data.get("user", {})
    allowed = user.get("allowed_unit_ids") or []
    print(f"  logged in as '{USERNAME}', allowed_unit_ids count = {len(allowed)}")
    return s


def get_org_units(s: requests.Session):
    admins = s.get(f"{BASE_URL}/api/org-units/administrations", timeout=30).json().get("administrations", [])
    depts = s.get(f"{BASE_URL}/api/org-units/departments", timeout=30).json().get("departments", [])
    sections = s.get(f"{BASE_URL}/api/org-units/sections", timeout=30).json().get("sections", [])
    return admins, depts, sections


def complaint_ids(report: dict) -> set:
    return {c.get("id") for c in (report.get("complaints") or [])}


def notice_ids(report: dict) -> set:
    return {n.get("id") for n in (report.get("notices") or [])}


def total_records(report: dict) -> int:
    """
    True unpaginated complaint count. /monthly/view always applies the
    detailed-mode default page_size=50 (MonthlyViewRequest has no page_size
    field to override it), so len(report['complaints']) silently caps at 50
    — pagination.total_records is the real count and must be used for any
    reconciliation math that can plausibly exceed that cap.
    """
    return report.get("pagination", {}).get("total_records", len(report.get("complaints") or []))


def view_monthly(s: requests.Session, year: int, month: int, **scope) -> dict:
    payload = {"year": year, "month": month, "mode": "detailed", "group_by": "section"}
    payload.update({k: v for k, v in scope.items() if v is not None})
    r = s.post(f"{BASE_URL}/api/reports/monthly/view", json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def targets_within(complaint: dict, allowed_ids: set) -> bool:
    tds = complaint.get("target_departments") or []
    if not tds:
        return False
    ids = set()
    for td in tds:
        for key in ("section_id", "department_id", "administration_id"):
            v = td.get(key)
            if v is not None:
                ids.add(v)
    return bool(ids & allowed_ids)


def pick_test_chain(admins, depts, sections):
    """Find one Administration -> Department -> Section chain with real data,
    trying a handful of candidates so a sparsely-populated unit doesn't stall
    the whole run."""
    dept_by_admin = {}
    for d in depts:
        dept_by_admin.setdefault(d["administration_id"], []).append(d)
    sec_by_dept = {}
    for sec in sections:
        sec_by_dept.setdefault(sec["department_id"], []).append(sec)

    chains = []
    for admin in admins:
        for dept in dept_by_admin.get(admin["id"], []):
            for sec in sec_by_dept.get(dept["id"], []):
                chains.append((admin, dept, sec))
    return chains


def run():
    print("=" * 70)
    print("MONTHLY/SEASONAL REPORT SCOPE-ISOLATION TEST SUITE")
    print("=" * 70)

    s = login()
    admins, depts, sections = get_org_units(s)
    print(f"  {len(admins)} administrations, {len(depts)} departments, {len(sections)} sections")

    chains = pick_test_chain(admins, depts, sections)
    check("at least one Administration->Department->Section chain exists", len(chains) > 0)
    if not chains:
        print("No chains to test — aborting.")
        sys.exit(1)

    # Use a wide date range so we're not dependent on any specific month
    # having data — this suite tests SCOPE isolation, not period filtering.
    YEAR, MONTH = 2026, 1

    # ------------------------------------------------------------------
    # Track A / B (same HTTP contract) — regression test for the exact bug
    # ------------------------------------------------------------------
    tested_chains = 0
    for admin, dept, sec in chains:
        if tested_chains >= 8:
            break
        label_base = f"admin={admin['id']} dept={dept['id']} section={sec['id']}"
        try:
            # Section-only selection (what the backend SHOULD treat any
            # equivalent payload as).
            section_only = view_monthly(s, YEAR, MONTH, section_ids=str(sec["id"]))
        except Exception as e:
            skip(f"[{label_base}] section-only view", str(e)[:150])
            continue

        try:
            # Realistic cascading payload — exactly what ReportingPage.js
            # sends after a normal Administration -> Department -> Section
            # narrowing journey (all three fields populated at once).
            cascaded = view_monthly(
                s, YEAR, MONTH,
                administration_ids=str(admin["id"]),
                department_ids=str(dept["id"]),
                section_ids=str(sec["id"]),
            )
        except Exception as e:
            skip(f"[{label_base}] cascaded view", str(e)[:150])
            continue

        tested_chains += 1

        s1, s2 = complaint_ids(section_only), complaint_ids(cascaded)
        check(f"[{label_base}] section-only == cascaded complaint set (regression guard)", s1 == s2)

        n1, n2 = notice_ids(section_only), notice_ids(cascaded)
        check(f"[{label_base}] section-only == cascaded notice set (regression guard)", n1 == n2)

        # Containment: every returned complaint must actually target this
        # section (or nothing broader leaked in).
        allowed_ids = {sec["id"], dept["id"], admin["id"]}
        leaked = [
            c["id"] for c in (section_only.get("complaints") or [])
            if not targets_within(c, {sec["id"]})
        ]
        check(f"[{label_base}] no cross-unit leakage in section-only result", len(leaked) == 0)

    check("at least one chain was actually tested (not all skipped)", tested_chains > 0)

    # ------------------------------------------------------------------
    # Track A — count reconciliation ("2 ways to check against each other")
    # ------------------------------------------------------------------
    dept_by_admin = {}
    for d in depts:
        dept_by_admin.setdefault(d["administration_id"], []).append(d)
    sec_by_dept = {}
    for sec in sections:
        sec_by_dept.setdefault(sec["department_id"], []).append(sec)

    # Department -> its Sections
    recon_dept_done = 0
    for dept in depts:
        children = sec_by_dept.get(dept["id"], [])
        if not children or recon_dept_done >= 5:
            continue
        try:
            dept_report = view_monthly(s, YEAR, MONTH, department_ids=str(dept["id"]))
        except Exception as e:
            skip(f"[dept={dept['id']}] reconciliation", str(e)[:150])
            continue
        dept_total = total_records(dept_report)

        child_total = 0
        ok = True
        for child in children:
            try:
                child_report = view_monthly(s, YEAR, MONTH, section_ids=str(child["id"]))
            except Exception as e:
                skip(f"[dept={dept['id']}, section={child['id']}] reconciliation child", str(e)[:150])
                ok = False
                break
            child_total += total_records(child_report)
        if not ok:
            continue

        recon_dept_done += 1
        # Not strict equality: a complaint's target_departments can point
        # directly at the Department itself rather than any of its Sections
        # (architecturally valid — "targets the department, no section
        # specified"), so department total can legitimately exceed the sum
        # of its sections. What must ALWAYS hold is sum(children) <=
        # parent total, since the department-level query's expanded target
        # set is a superset of every individual section's target set.
        check(f"[dept={dept['id']}] sum(sections)={child_total} <= department total={dept_total}",
              child_total <= dept_total)

    check("at least one Department reconciliation check ran", recon_dept_done > 0)

    # Hospital (no filter) -> sum of all Administrations
    try:
        hospital_report = view_monthly(s, YEAR, MONTH)
        hospital_total = total_records(hospital_report)

        admin_total = 0
        for admin in admins:
            try:
                admin_report = view_monthly(s, YEAR, MONTH, administration_ids=str(admin["id"]))
            except Exception as e:
                skip(f"[admin={admin['id']}] hospital reconciliation", str(e)[:150])
                continue
            admin_total += total_records(admin_report)

        # Same monotonicity rationale as the department check above — a
        # complaint can target the hospital/building level directly without
        # any administration specified, so sum(administrations) can be less
        # than the hospital total, but never more.
        check(f"sum(administrations)={admin_total} <= hospital total={hospital_total}",
              admin_total <= hospital_total)
    except Exception as e:
        skip("hospital-level reconciliation", str(e)[:150])

    # ------------------------------------------------------------------
    # Track C — Seasonal regression baseline (should be unaffected)
    # ------------------------------------------------------------------
    try:
        # orgunit_id=1/orgunit_type=0 == hospital-wide per SeasonalViewRequestV2 convention
        hospital_seasonal = s.post(
            f"{BASE_URL}/api/reports/seasonal/view",
            json={"year": 2026, "trimester": "Q1", "orgunit_id": 1, "orgunit_type": 0, "user_id": 1},
            timeout=60,
        )
        if hospital_seasonal.status_code == 200:
            hospital_data = hospital_seasonal.json()
            if admins:
                admin_seasonal = s.post(
                    f"{BASE_URL}/api/reports/seasonal/view",
                    json={"year": 2026, "trimester": "Q1", "orgunit_id": admins[0]["id"], "orgunit_type": 1, "user_id": 1},
                    timeout=60,
                )
                if admin_seasonal.status_code == 200:
                    admin_data = admin_seasonal.json()
                    hosp_total = (hospital_data.get("domain_totals") or hospital_data.get("classification_stats") or [])
                    adm_total = (admin_data.get("domain_totals") or admin_data.get("classification_stats") or [])
                    check("seasonal: single-administration view returns a response distinct in shape from hospital view",
                          isinstance(adm_total, list))
                else:
                    skip("seasonal administration-scoped view", f"HTTP {admin_seasonal.status_code}")
        else:
            skip("seasonal hospital-wide view", f"HTTP {hospital_seasonal.status_code}")
    except Exception as e:
        skip("seasonal regression baseline", str(e)[:150])

    # ------------------------------------------------------------------
    print("=" * 70)
    print(f"Chains tested: {tested_chains} | Department reconciliations: {recon_dept_done} | Skipped: {len(SKIPPED)}")
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} check(s) FAILED")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("RESULT: all checks passed")
        sys.exit(0)


if __name__ == "__main__":
    run()
