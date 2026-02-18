"""
PHASE M — FUNCTIONAL CONNECTION TEST
Verify the centralized connection can perform real database operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from core.database import get_connection

print("=" * 70)
print("PHASE M — FUNCTIONAL DATABASE CONNECTION TEST")
print("=" * 70)

try:
    # Test 1: Get connection
    print("\n✓ TEST 1: Get connection from core.database")
    conn = get_connection()
    print(f"  Connection established: {conn}")
    print("  ✅ PASS")
    
    # Test 2: Execute simple query
    print("\n✓ TEST 2: Execute SQL Server version query")
    cursor = conn.cursor()
    cursor.execute("SELECT @@VERSION")
    version = cursor.fetchone()[0]
    print(f"  SQL Server: {version[:80]}")
    print("  ✅ PASS")
    
    # Test 3: Query actual database
    print("\n✓ TEST 3: Query database name")
    cursor.execute("SELECT DB_NAME()")
    db_name = cursor.fetchone()[0]
    print(f"  Connected to database: {db_name}")
    assert db_name == "IncidentManager", f"Expected 'IncidentManager', got '{db_name}'"
    print("  ✅ PASS")
    
    # Test 4: Query a sample table
    print("\n✓ TEST 4: Query sample table (APP_IncidentCase)")
    cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
    count = cursor.fetchone()[0]
    print(f"  Total incident cases: {count}")
    print("  ✅ PASS")
    
    # Test 5: Close connection
    print("\n✓ TEST 5: Close connection properly")
    conn.close()
    print("  Connection closed successfully")
    print("  ✅ PASS")
    
    print("\n" + "=" * 70)
    print("✅ ALL FUNCTIONAL TESTS PASSED")
    print("=" * 70)
    print("\n🎯 Core database module is working correctly!")
    print("   - Connection established")
    print("   - Queries execute successfully")
    print("   - Connected to correct database")
    print("   - Can access production tables")
    print("   - Connection closes cleanly")
    
except Exception as e:
    print(f"\n❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
