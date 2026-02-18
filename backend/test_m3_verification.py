"""
PHASE M — TEST M3 — VERIFY API DB_LAYER CLEANUP
Verify backend/api/db_layer modules no longer define local get_connection functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("PHASE M — TEST M3 — API DB_LAYER CLEANUP VERIFICATION")
print("=" * 70)

# =====================================================================
# SCAN FOLDER: backend/api/db_layer/
# =====================================================================

import glob

db_layer_files = glob.glob("api/db_layer/**/*.py", recursive=True)
db_layer_files = [f for f in db_layer_files if not f.endswith('__init__.py')]

print(f"\n✓ Files scanned: {len(db_layer_files)}")
for f in sorted(db_layer_files):
    print(f"  - {f}")

# =====================================================================
# CHECK 1: Search for local get_connection definitions
# =====================================================================

print("\n✓ CHECK 1: Search for 'def get_connection(' definitions")
violations_get_connection = []

for filepath in db_layer_files:
    # Skip training_db.py (uses sqlite3)
    if 'training_db.py' in filepath:
        print(f"  [SKIP] {filepath} (sqlite3 file - excluded)")
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'def get_connection(' in content:
            violations_get_connection.append(filepath)
            print(f"  ❌ VIOLATION: {filepath}")

if not violations_get_connection:
    print("  ✅ PASS: ZERO local get_connection() definitions found")
else:
    print(f"  ❌ FAIL: Found {len(violations_get_connection)} files with local get_connection()")

# =====================================================================
# CHECK 2: Search for import from core.database
# =====================================================================

print("\n✓ CHECK 2: Search for 'from core.database import get_connection'")
files_with_import = []
files_without_import = []

for filepath in db_layer_files:
    # Skip training_db.py and __init__.py
    if 'training_db.py' in filepath or '__init__.py' in filepath:
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'from core.database import get_connection' in content:
            files_with_import.append(filepath)
        else:
            # Check if file actually uses get_connection
            if 'get_connection()' in content:
                files_without_import.append(filepath)
                print(f"  ⚠️  WARNING: {filepath} uses get_connection but has no import")

print(f"  Files with correct import: {len(files_with_import)}")
if files_without_import:
    print(f"  ❌ Files missing import: {len(files_without_import)}")
    for f in files_without_import:
        print(f"    - {f}")
else:
    print("  ✅ PASS: All files that use get_connection() have the import")

# =====================================================================
# CHECK 3: Search for direct pyodbc.connect() calls
# =====================================================================

print("\n✓ CHECK 3: Search for direct 'pyodbc.connect(' calls")
violations_pyodbc = []

for filepath in db_layer_files:
    # Skip training_db.py (uses sqlite3)
    if 'training_db.py' in filepath:
        continue
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'pyodbc.connect(' in content:
            violations_pyodbc.append(filepath)
            print(f"  ❌ VIOLATION: {filepath}")

if not violations_pyodbc:
    print("  ✅ PASS: ZERO direct pyodbc.connect() calls found")
else:
    print(f"  ❌ FAIL: Found {len(violations_pyodbc)} files with direct pyodbc.connect()")

# =====================================================================
# SQLITE FILES REPORT
# =====================================================================

print("\n✓ SQLITE FILES (Excluded from checks):")
sqlite_files = [f for f in db_layer_files if 'training_db.py' in f]
if sqlite_files:
    for f in sqlite_files:
        print(f"  - {f} (excluded - uses sqlite3)")
else:
    print("  None found in api/db_layer/")

# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
total_violations = len(violations_get_connection) + len(violations_pyodbc) + len(files_without_import)

if total_violations == 0:
    print("✅ ALL CHECKS PASSED - M3 VERIFICATION: SUCCESS")
    print("=" * 70)
    print("\n📊 SUMMARY:")
    print(f"  ✓ Files scanned: {len(db_layer_files)}")
    print(f"  ✓ Local get_connection() definitions: 0")
    print(f"  ✓ Direct pyodbc.connect() calls: 0")
    print(f"  ✓ Files with correct imports: {len(files_with_import)}")
    print(f"  ✓ Files excluded (sqlite): {len(sqlite_files)}")
    print("\n🎯 M-3 VERIFICATION: COMPLETE\n")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("=" * 70)
    print(f"\n⚠️  Total violations: {total_violations}")
    print(f"  - Local get_connection() definitions: {len(violations_get_connection)}")
    print(f"  - Direct pyodbc.connect() calls: {len(violations_pyodbc)}")
    print(f"  - Missing imports: {len(files_without_import)}")
    print("\n🔴 M-3 VERIFICATION: FAILED\n")
    sys.exit(1)
