"""
PHASE M — TEST M5 — STATIC VERIFICATION SCAN
Scan entire backend tree for remaining connection violations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("PHASE M — TEST M5 — STATIC VERIFICATION SCAN")
print("=" * 70)

# =====================================================================
# SCAN: All .py files in backend/ (excluding SQLite and test files)
# =====================================================================

import glob

all_py_files = glob.glob("**/*.py", recursive=True)

# Exclusions
excluded_patterns = [
    '__pycache__',
    'venv',
    'env',
    'node_modules',
    '.git',
    'training_db.py',  # SQLite, as per Q4 answer
    'test_',           # Test files
    '__init__.py',     # Usually empty
    'rename_get_db_connection.py'  # Utility script with false positive
]

production_files = []
for f in all_py_files:
    skip = False
    for pattern in excluded_patterns:
        if pattern in f:
            skip = True
            break
    if not skip:
        production_files.append(f)

print(f"\n✓ Production Python files scanned: {len(production_files)}")

# =====================================================================
# CHECK 1: Search for local get_connection definitions (exclude core.database)
# =====================================================================

print("\n✓ CHECK 1: Search for 'def get_connection(' (excluding core/database.py)")
violations_get_connection = []

for filepath in production_files:
    if filepath == 'core\\database.py' or filepath == 'core/database.py':
        continue  # This is the canonical module
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'def get_connection(' in content:
            violations_get_connection.append(filepath)
            print(f"  ❌ VIOLATION: {filepath}")

if not violations_get_connection:
    print("  ✅ PASS: ZERO unauthorized get_connection() definitions found")
else:
    print(f"  ❌ FAIL: Found {len(violations_get_connection)} violations")

# =====================================================================
# CHECK 2: Search for get_db_connection definitions
# =====================================================================

print("\n✓ CHECK 2: Search for 'def get_db_connection('")
violations_get_db_connection = []

for filepath in production_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'def get_db_connection(' in content:
            violations_get_db_connection.append(filepath)
            print(f"  ❌ VIOLATION: {filepath}")

if not violations_get_db_connection:
    print("  ✅ PASS: ZERO get_db_connection() definitions found")
else:
    print(f"  ❌ FAIL: Found {len(violations_get_db_connection)} violations")

# =====================================================================
# CHECK 3: Search for direct pyodbc.connect() (exclude core.database)
# =====================================================================

print("\n✓ CHECK 3: Search for 'pyodbc.connect(' (excluding core/database.py)")
violations_pyodbc = []

for filepath in production_files:
    if filepath == 'core\\database.py' or filepath == 'core/database.py':
        continue  # Canonical module is allowed to use pyodbc.connect
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'pyodbc.connect(' in content:
            violations_pyodbc.append(filepath)
            print(f"  ❌ VIOLATION: {filepath}")

if not violations_pyodbc:
    print("  ✅ PASS: ZERO unauthorized pyodbc.connect() calls found")
else:
    print(f"  ❌ FAIL: Found {len(violations_pyodbc)} violations")

# CHECK 4: Verify files using connections import from core
# =====================================================================

print("\n✓ CHECK 4: Analysis of connection usage patterns")

files_using_get_connection = []
files_missing_import = []

for filepath in production_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Check if file calls get_connection()
        if 'get_connection()' in content:
            files_using_get_connection.append(filepath)
            
            # Verify it has the import (allow both absolute and relative imports)
            has_import = (
                'from core.database import get_connection' in content or
                'from backend.core.database import get_connection' in content
            )
            
            # Skip core/database.py itself (it defines the function)
            is_core_database = (filepath == 'core\\database.py' or filepath == 'core/database.py')
            
            if not has_import and not is_core_database:
                files_missing_import.append(filepath)

print(f"  Files calling get_connection(): {len(files_using_get_connection)}")
print(f"  Files with correct import: {len(files_using_get_connection) - len(files_missing_import)}")

if files_missing_import:
    print(f"  ❌ Files missing import: {len(files_missing_import)}")
    for f in files_missing_import:
        print(f"    - {f}")
else:
    print("  ✅ PASS: All files using get_connection() have the correct import")

# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
total_violations = (len(violations_get_connection) + len(violations_get_db_connection) +
                   len(violations_pyodbc) + len(files_missing_import))

if total_violations == 0:
    print("✅ ALL CHECKS PASSED - M5 STATIC SCAN: SUCCESS")
    print("=" * 70)
    print("\n📊 SUMMARY:")
    print(f"  ✓ Production files scanned: {len(production_files)}")
    print(f"  ✓ Unauthorized get_connection() definitions: 0")
    print(f"  ✓ get_db_connection() definitions: 0")
    print(f"  ✓ Unauthorized pyodbc.connect() calls: 0")
    print(f"  ✓ Files using get_connection(): {len(files_using_get_connection)}")
    print(f"  ✓ Missing imports: 0")
    print("\n🎯 M-5 STATIC VERIFICATION: COMPLETE")
    print("✅ PHASE M CENTRALIZATION: 100% COMPLIANT\n")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("=" * 70)
    print(f"\n⚠️  Total violations: {total_violations}")
    print(f"  - Unauthorized get_connection() definitions: {len(violations_get_connection)}")
    print(f"  - get_db_connection() definitions: {len(violations_get_db_connection)}")
    print(f"  - Unauthorized pyodbc.connect() calls: {len(violations_pyodbc)}")
    print(f"  - Missing imports: {len(files_missing_import)}")
    print("\n🔴 M-5 STATIC VERIFICATION: FAILED\n")
    sys.exit(1)
