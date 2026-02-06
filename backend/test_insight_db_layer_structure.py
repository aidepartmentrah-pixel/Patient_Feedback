"""
Test Insight DB Layer Structure
Verifies that insight_db.py module is correctly structured with no import/naming errors.

Run: python backend/test_insight_db_layer_structure.py
"""

import sys
import os
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

print("=" * 80)
print("INSIGHT DB LAYER STRUCTURE TEST")
print("=" * 80)

# Test 1: Import module
print("\n[TEST 1] Import insight_db module...")
try:
    from api_v2.db_layer import insight_db
    print("✅ SUCCESS: Module imported successfully")
except Exception as e:
    print(f"❌ FAILURE: Could not import module")
    print(f"   Error: {e}")
    sys.exit(1)

# Test 2: Check get_db_connection exists
print("\n[TEST 2] Check get_db_connection function exists...")
try:
    assert hasattr(insight_db, 'get_db_connection')
    assert callable(insight_db.get_db_connection)
    print("✅ SUCCESS: get_db_connection exists and is callable")
except Exception as e:
    print(f"❌ FAILURE: {e}")
    sys.exit(1)

# Test 3: Check all required functions exist
print("\n[TEST 3] Check all required functions exist...")
required_functions = [
    'get_subcase_status_counts',
    'get_action_item_counts',
    'get_stuck_subcases',
    'get_subcase_created_time_buckets'
]

missing_functions = []
for func_name in required_functions:
    if not hasattr(insight_db, func_name):
        missing_functions.append(func_name)
    elif not callable(getattr(insight_db, func_name)):
        missing_functions.append(f"{func_name} (not callable)")

if missing_functions:
    print(f"❌ FAILURE: Missing or invalid functions:")
    for func in missing_functions:
        print(f"   - {func}")
    sys.exit(1)
else:
    print(f"✅ SUCCESS: All {len(required_functions)} required functions exist")
    for func_name in required_functions:
        print(f"   ✓ {func_name}")

# Test 4: Check function signatures
print("\n[TEST 4] Check function signatures (parameter names)...")
import inspect

try:
    # get_subcase_status_counts
    sig = inspect.signature(insight_db.get_subcase_status_counts)
    params = list(sig.parameters.keys())
    assert 'conn' in params, "get_subcase_status_counts missing 'conn' parameter"
    assert 'allowed_unit_ids' in params, "get_subcase_status_counts missing 'allowed_unit_ids' parameter"
    print("   ✓ get_subcase_status_counts(conn, allowed_unit_ids)")
    
    # get_action_item_counts
    sig = inspect.signature(insight_db.get_action_item_counts)
    params = list(sig.parameters.keys())
    assert 'conn' in params, "get_action_item_counts missing 'conn' parameter"
    assert 'allowed_unit_ids' in params, "get_action_item_counts missing 'allowed_unit_ids' parameter"
    print("   ✓ get_action_item_counts(conn, allowed_unit_ids)")
    
    # get_stuck_subcases
    sig = inspect.signature(insight_db.get_stuck_subcases)
    params = list(sig.parameters.keys())
    assert 'conn' in params, "get_stuck_subcases missing 'conn' parameter"
    assert 'allowed_unit_ids' in params, "get_stuck_subcases missing 'allowed_unit_ids' parameter"
    assert 'days_threshold' in params, "get_stuck_subcases missing 'days_threshold' parameter"
    print("   ✓ get_stuck_subcases(conn, allowed_unit_ids, days_threshold)")
    
    # get_subcase_created_time_buckets
    sig = inspect.signature(insight_db.get_subcase_created_time_buckets)
    params = list(sig.parameters.keys())
    assert 'conn' in params, "get_subcase_created_time_buckets missing 'conn' parameter"
    assert 'allowed_unit_ids' in params, "get_subcase_created_time_buckets missing 'allowed_unit_ids' parameter"
    assert 'bucket' in params, "get_subcase_created_time_buckets missing 'bucket' parameter"
    print("   ✓ get_subcase_created_time_buckets(conn, allowed_unit_ids, bucket)")
    
    print("✅ SUCCESS: All function signatures are correct")
except AssertionError as e:
    print(f"❌ FAILURE: {e}")
    sys.exit(1)

# Test 5: Check docstrings exist
print("\n[TEST 5] Check all functions have docstrings...")
functions_without_docs = []
for func_name in required_functions:
    func = getattr(insight_db, func_name)
    if not func.__doc__ or len(func.__doc__.strip()) < 10:
        functions_without_docs.append(func_name)

if functions_without_docs:
    print(f"❌ FAILURE: Functions missing docstrings:")
    for func in functions_without_docs:
        print(f"   - {func}")
    sys.exit(1)
else:
    print(f"✅ SUCCESS: All functions have docstrings")

# Test 6: Try to get a database connection (will fail if DB not available, but tests connection logic)
print("\n[TEST 6] Test database connection function (structure only)...")
try:
    # Just check that the function can be called (won't actually connect in test)
    connection_func = insight_db.get_db_connection
    print(f"✅ SUCCESS: Connection function is callable")
    print(f"   Note: Actual DB connection not tested (requires live database)")
except Exception as e:
    print(f"❌ FAILURE: Connection function issue: {e}")
    sys.exit(1)

# Test 7: Check imports in module
print("\n[TEST 7] Check required imports...")
try:
    assert hasattr(insight_db, 'Dict'), "Missing typing.Dict import"
    assert hasattr(insight_db, 'Any'), "Missing typing.Any import"
    assert hasattr(insight_db, 'List'), "Missing typing.List import"
    assert hasattr(insight_db, 'Optional'), "Missing typing.Optional import"
    assert hasattr(insight_db, 'datetime'), "Missing datetime import"
    assert hasattr(insight_db, 'pyodbc'), "Missing pyodbc import"
    print("✅ SUCCESS: All required imports present")
    print("   ✓ typing (Dict, Any, List, Optional)")
    print("   ✓ datetime")
    print("   ✓ pyodbc")
except AssertionError as e:
    print(f"❌ FAILURE: {e}")
    sys.exit(1)

# Final summary
print("\n" + "=" * 80)
print("✅ ALL STRUCTURE TESTS PASSED")
print("=" * 80)
print("\nModule Status:")
print("  • Module imports successfully")
print("  • All 4 functions defined with correct signatures")
print("  • All functions have comprehensive docstrings")
print("  • Database connection helper exists")
print("  • All required imports present")
print("\nNext Step: Implement SQL queries in each function")
print("=" * 80)
