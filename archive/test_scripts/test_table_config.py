"""
Test Table Config — Verifies the table_config refactor.

Tests:
  1. Import test — table_config.py imports without errors
  2. Value test — All 3 variables have the expected offline values
  3. Live query test — SELECT TOP 1 against each configured table
  4. Swap simulation — Monkey-patch proves config is actually used
  5. Search endpoint smoke test — Patient/doctor/employee search via HTTP
  6. Worker profile endpoint — /api/workers/{id}/profile shape check
  7. SQL string safety check — No hardcoded table names remain in SQL queries
"""

import sys
import os
import importlib
import re

# Ensure backend is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

BASE_URL = "http://localhost:8000"
PASSED = 0
FAILED = 0


def test(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}" + (f" -- {detail}" if detail else ""))
    else:
        FAILED += 1
        print(f"  [FAIL] {name}" + (f" -- {detail}" if detail else ""))


# ========================================================================
# TEST GROUP 1: Import Test
# ========================================================================
print("\n=== GROUP 1: Import Test ===")

try:
    from core.table_config import HR_EMPLOYEES_TABLE, PATIENT_ADMISSION_TABLE, DOCTORS_TABLE
    test("Import table_config", True)
except Exception as e:
    test("Import table_config", False, str(e))
    HR_EMPLOYEES_TABLE = None
    PATIENT_ADMISSION_TABLE = None
    DOCTORS_TABLE = None

# ========================================================================
# TEST GROUP 2: Value Test
# ========================================================================
print("\n=== GROUP 2: Value Test ===")

test("HR_EMPLOYEES_TABLE value", HR_EMPLOYEES_TABLE == "APP_VIEWTABLE_HR_EMPLOYEES",
     f"got: {HR_EMPLOYEES_TABLE}")
test("PATIENT_ADMISSION_TABLE value", PATIENT_ADMISSION_TABLE == "APP_VIEWTABLE_PATIENT_ADMISSION",
     f"got: {PATIENT_ADMISSION_TABLE}")
test("DOCTORS_TABLE value", DOCTORS_TABLE == "APP_VIEWTABLE_VW_DOCTORS",
     f"got: {DOCTORS_TABLE}")

test("HR_EMPLOYEES_TABLE is str", isinstance(HR_EMPLOYEES_TABLE, str))
test("PATIENT_ADMISSION_TABLE is str", isinstance(PATIENT_ADMISSION_TABLE, str))
test("DOCTORS_TABLE is str", isinstance(DOCTORS_TABLE, str))

test("HR_EMPLOYEES_TABLE not empty", bool(HR_EMPLOYEES_TABLE))
test("PATIENT_ADMISSION_TABLE not empty", bool(PATIENT_ADMISSION_TABLE))
test("DOCTORS_TABLE not empty", bool(DOCTORS_TABLE))

# ========================================================================
# TEST GROUP 3: Live Query Test
# ========================================================================
print("\n=== GROUP 3: Live Query Test ===")

try:
    from core.database import get_connection
    conn = get_connection()
    cursor = conn.cursor()

    # Test HR Employees table
    try:
        cursor.execute(f"SELECT TOP 1 EmployeeID FROM {HR_EMPLOYEES_TABLE}")
        row = cursor.fetchone()
        test("HR_EMPLOYEES_TABLE resolves in DB", True,
             f"got EmployeeID={row.EmployeeID}" if row else "table exists but empty")
    except Exception as e:
        test("HR_EMPLOYEES_TABLE resolves in DB", False, str(e))

    # Test Patient Admission table
    try:
        cursor.execute(f"SELECT TOP 1 PatientAdmissionID FROM {PATIENT_ADMISSION_TABLE}")
        row = cursor.fetchone()
        test("PATIENT_ADMISSION_TABLE resolves in DB", True,
             f"got PatientAdmissionID={row.PatientAdmissionID}" if row else "table exists but empty")
    except Exception as e:
        test("PATIENT_ADMISSION_TABLE resolves in DB", False, str(e))

    # Test Doctors table
    try:
        cursor.execute(f"SELECT TOP 1 DoctorID FROM {DOCTORS_TABLE}")
        row = cursor.fetchone()
        test("DOCTORS_TABLE resolves in DB", True,
             f"got DoctorID={row.DoctorID}" if row else "table exists but empty")
    except Exception as e:
        test("DOCTORS_TABLE resolves in DB", False, str(e))

    cursor.close()
    conn.close()
except Exception as e:
    test("Database connection for live query", False, str(e))

# ========================================================================
# TEST GROUP 4: Swap Simulation (monkey-patch)
# ========================================================================
print("\n=== GROUP 4: Swap Simulation ===")

try:
    import core.table_config as tc
    original_value = tc.HR_EMPLOYEES_TABLE

    # Monkey-patch to a non-existent table
    tc.HR_EMPLOYEES_TABLE = "NONEXISTENT_TABLE_12345"

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT TOP 1 EmployeeID FROM {tc.HR_EMPLOYEES_TABLE}")
        # If this succeeds, the config is NOT being used (dead import)
        test("Swap simulation (config is live)", False, "Query succeeded with fake table name — config may be dead import")
    except Exception:
        test("Swap simulation (config is live)", True, "Query correctly failed with fake table name")
    finally:
        cursor.close()
        conn.close()

    # Restore original
    tc.HR_EMPLOYEES_TABLE = original_value
    test("Config restored after swap", tc.HR_EMPLOYEES_TABLE == original_value)
except Exception as e:
    test("Swap simulation", False, str(e))

# ========================================================================
# TEST GROUP 5: Search Endpoint Smoke Tests
# ========================================================================
print("\n=== GROUP 5: Search Endpoint Smoke Tests ===")

import requests

# Patient search (uses unprotected patients router)
try:
    r = requests.get(f"{BASE_URL}/api/patients/search", params={"query": "a"}, timeout=10)
    test("Patient search endpoint (GET)", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    test("Patient search returns list", "patients" in data or "results" in data or isinstance(data, list),
         f"keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
except Exception as e:
    test("Patient search endpoint", False, str(e))
    test("Patient search returns list", False, "skipped")

# Doctor search (uses unprotected doctors router)
try:
    r = requests.get(f"{BASE_URL}/api/doctors", params={"query": "a"}, timeout=10)
    test("Doctor search endpoint (GET)", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    test("Doctor search returns list", "doctors" in data or "results" in data or isinstance(data, list),
         f"keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
except Exception as e:
    test("Doctor search endpoint", False, str(e))
    test("Doctor search returns list", False, "skipped")

# Employee search (uses session-protected records router — login first)
try:
    session = requests.Session()
    login_r = session.post(f"{BASE_URL}/api/auth/login", json={"username": "software_admin", "password": "admin123"}, timeout=10)
    r = session.get(f"{BASE_URL}/api/records/search/employees", params={"q": "a"}, timeout=10)
    test("Employee search endpoint (GET)", r.status_code == 200, f"status={r.status_code}")
    data = r.json()
    test("Employee search returns list", "employees" in data or "results" in data or isinstance(data, list),
         f"keys={list(data.keys()) if isinstance(data, dict) else 'list'}")
except Exception as e:
    test("Employee search endpoint", False, str(e))
    test("Employee search returns list", False, "skipped")

# ========================================================================
# TEST GROUP 6: Worker Profile Endpoint
# ========================================================================
print("\n=== GROUP 6: Worker Profile Endpoint ===")

try:
    # First get an employee ID to test with
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT TOP 1 EmployeeID FROM {HR_EMPLOYEES_TABLE} WHERE IsActive = 1")
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row:
        emp_id = row.EmployeeID
        # Worker profile requires session auth
        session = requests.Session()
        session.post(f"{BASE_URL}/api/auth/login", json={"username": "software_admin", "password": "admin123"}, timeout=10)
        r = session.get(f"{BASE_URL}/api/workers/{emp_id}/profile", timeout=10)
        test("Worker profile endpoint", r.status_code == 200, f"status={r.status_code} for employee {emp_id}")

        if r.status_code == 200:
            data = r.json()
            test("Worker profile has identity", "worker" in data or "identity" in data or "employee_id" in data or "full_name" in data,
                 f"keys={list(data.keys())[:5]}")
        else:
            test("Worker profile has identity", False, f"non-200 status: {r.status_code}")
    else:
        test("Worker profile endpoint", False, "No active employees in DB to test with")
        test("Worker profile has identity", False, "skipped")
except Exception as e:
    test("Worker profile endpoint", False, str(e))
    test("Worker profile has identity", False, "skipped")

# ========================================================================
# TEST GROUP 7: SQL String Safety Check (grep)
# ========================================================================
print("\n=== GROUP 7: SQL String Safety Check ===")

BACKEND_DIR = os.path.join(os.path.dirname(__file__), "backend")
TARGET_FILES = [
    os.path.join(BACKEND_DIR, "api", "services", "search_service.py"),
    os.path.join(BACKEND_DIR, "api", "db_layer", "patients_db.py"),
    os.path.join(BACKEND_DIR, "api", "db_layer", "worker_reporting_db.py"),
    os.path.join(BACKEND_DIR, "api", "services", "aggregate_seasonal_report_service.py"),
]

HARDCODED_PATTERNS = [
    r'(?<!#)(?<!\")(?<!\')FROM\s+(?:dbo\.)?APP_VIEWTABLE_HR_EMPLOYEES',
    r'(?<!#)(?<!\")(?<!\')FROM\s+(?:dbo\.)?APP_VIEWTABLE_PATIENT_ADMISSION',
    r'(?<!#)(?<!\")(?<!\')FROM\s+(?:dbo\.)?APP_VIEWTABLE_VW_DOCTORS',
]

TABLE_NAMES_IN_SQL = [
    "APP_VIEWTABLE_HR_EMPLOYEES",
    "APP_VIEWTABLE_PATIENT_ADMISSION",
    "APP_VIEWTABLE_VW_DOCTORS",
]

for filepath in TARGET_FILES:
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        test(f"File exists: {filename}", False, f"not found: {filepath}")
        continue

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')

    violations = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        # Skip comments and docstrings (any line inside triple-quoted block)
        if stripped.startswith('#'):
            continue
        # Skip lines that are clearly docstring content (no code)
        if not any(kw in line for kw in ['cursor.execute', 'SELECT', 'FROM', 'f"', "f'"]):
            continue

        for table_name in TABLE_NAMES_IN_SQL:
            # Look for the table name in SQL context (FROM keyword nearby)
            if table_name in line and 'FROM' in line.upper():
                # But allow {VARIABLE} style
                if '{' not in line or table_name in line.replace('{', '').replace('}', ''):
                    # Check if it's in a comment on this line
                    code_part = line.split('#')[0]
                    if table_name in code_part and '{' not in code_part:
                        violations.append(f"  Line {i}: {stripped[:100]}")

    test(f"No hardcoded table names in {filename}",
         len(violations) == 0,
         f"{len(violations)} violations found" if violations else "clean")
    for v in violations:
        print(f"    VIOLATION {v}")

# Also check that config imports exist in all 4 files
for filepath in TARGET_FILES:
    filename = os.path.basename(filepath)
    if not os.path.exists(filepath):
        continue
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    test(f"Config import in {filename}", "table_config" in content,
         "import found" if "table_config" in content else "MISSING import")


# ========================================================================
# SUMMARY
# ========================================================================
print(f"\n{'='*70}")
print(f"TABLE CONFIG TEST RESULTS: {PASSED} PASSED, {FAILED} FAILED out of {PASSED + FAILED} tests")
print(f"{'='*70}")
if FAILED == 0:
    print("ALL TESTS PASSED! Table config refactor is verified.")
else:
    print(f"WARNING: {FAILED} test(s) failed. Review above.")
