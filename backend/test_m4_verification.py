"""
PHASE M — TEST M4 — VERIFY API_V2 DB_LAYER CLEANUP
Verify backend/api_v2/db_layer uses only core get_connection.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("=" * 70)
print("PHASE M — TEST M4 — API_V2 DB_LAYER CLEANUP VERIFICATION")
print("=" * 70)

# =====================================================================
# SCAN FOLDER: backend/api_v2/db_layer/
# =====================================================================

import glob

db_layer_files = glob.glob("api_v2/db_layer/**/*.py", recursive=True)
db_layer_files = [f for f in db_layer_files if not f.endswith('__init__.py')]

print(f"\n✓ Files scanned: {len(db_layer_files)}")
for f in sorted(db_layer_files):
    print(f"  - {f}")

# =====================================================================
# CHECK 1: Search for local get_connection/get_db_connection definitions
# =====================================================================

print("\n✓ CHECK 1: Search for local connection function definitions")
violations_definitions = []

for filepath in db_layer_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'def get_connection(' in content or 'def get_db_connection(' in content:
            violations_definitions.append(filepath)
            print(f"  ❌ VIOLATION: {filepath}")

if not violations_definitions:
    print("  ✅ PASS: ZERO local connection function definitions found")
else:
    print(f"  ❌ FAIL: Found {len(violations_definitions)} files with local definitions")

# =====================================================================
# CHECK 2: Search for direct pyodbc.connect() calls
# =====================================================================

print("\n✓ CHECK 2: Search for direct 'pyodbc.connect(' calls")
violations_pyodbc = []

for filepath in db_layer_files:
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
# CHECK 3: Search for core.database import
# =====================================================================

print("\n✓ CHECK 3: Search for 'from core.database import get_connection'")
files_with_import = []
files_without_import = []

for filepath in db_layer_files:
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
# CHECK 4: Verify get_db_connection() renamed to get_connection()
# =====================================================================

print("\n✓ CHECK 4: Check for remaining get_db_connection() calls")
violations_old_name = []

for filepath in db_layer_files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'get_db_connection()' in content:
            violations_old_name.append(filepath)
            print(f"  ❌ VIOLATION: {filepath} - still using get_db_connection()")

if not violations_old_name:
    print("  ✅ PASS: All uses renamed to get_connection()")
else:
    print(f"  ❌ FAIL: Found {len(violations_old_name)} files still using get_db_connection()")

# =====================================================================
# FINAL SUMMARY
# =====================================================================

print("\n" + "=" * 70)
total_violations = (len(violations_definitions) + len(violations_pyodbc) +
                   len(files_without_import) + len(violations_old_name))

if total_violations == 0:
    print("✅ ALL CHECKS PASSED - M4 VERIFICATION: SUCCESS")
    print("=" * 70)
    print("\n📊 SUMMARY:")
    print(f"  ✓ Files scanned: {len(db_layer_files)}")
    print(f"  ✓ Local connection definitions: 0")
    print(f"  ✓ Direct pyodbc.connect() calls: 0")
    print(f"  ✓ Files with correct imports: {len(files_with_import)}")
    print(f"  ✓ Old function name (get_db_connection) usage: 0")
    print("\n🎯 M-4 VERIFICATION: COMPLETE\n")
    sys.exit(0)
else:
    print("❌ VERIFICATION FAILED")
    print("=" * 70)
    print(f"\n⚠️  Total violations: {total_violations}")
    print(f"  - Local connection definitions: {len(violations_definitions)}")
    print(f"  - Direct pyodbc.connect() calls: {len(violations_pyodbc)}")
    print(f"  - Missing imports: {len(files_without_import)}")
    print(f"  - Old function name usage: {len(violations_old_name)}")
    print("\n🔴 M-4 VERIFICATION: FAILED\n")
    sys.exit(1)
