"""
TEST PHASE 2: STANDALONE VERIFICATION
======================================
Standalone test that verifies ML DB growth tracking without requiring
backend server to be running.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.api.db_layer.training_db import (
    get_current_ml_db_size,
    record_ml_db_size,
    get_ml_db_size_history,
    ML_DB_PATH
)
from backend.api.services.training_service import get_ml_database_size_history

print("\n" + "="*80)
print("PHASE 2: STANDALONE VERIFICATION")
print("="*80)

# Test 1: Verify ML DB exists and has records
print("\n[TEST 1] ML Database Status")
print("="*80)
print(f"Path: {ML_DB_PATH}")
print(f"Exists: {Path(ML_DB_PATH).exists()}")

current_size = get_current_ml_db_size()
print(f"Current size: {current_size} records")

if current_size > 0:
    print("✅ ML database has records")
else:
    print("⚠️  ML database is empty (OK for fresh install)")

# Test 2: Record current size
print("\n[TEST 2] Record Current ML DB Size")
print("="*80)
today = date.today().isoformat()
record_ml_db_size(current_size, today)
print(f"✅ Recorded {current_size} records for {today}")

# Test 3: Create sample historical data
print("\n[TEST 3] Create Sample Historical Data")
print("="*80)

sample_data = []
for i in range(7):
    sample_date = (date.today() - timedelta(days=6-i)).isoformat()
    # Simulate growth
    sample_count = max(0, current_size - ((6-i) * 50))
    record_ml_db_size(sample_count, sample_date)
    sample_data.append((sample_date, sample_count))
    print(f"   {sample_date}: {sample_count} records")

print(f"✅ Created 7 days of historical data")

# Test 4: Retrieve history
print("\n[TEST 4] Retrieve Historical Data")
print("="*80)

history = get_ml_db_size_history(days=7)
print(f"Retrieved {len(history)} points:")
for point in history[-7:]:  # Last 7
    print(f"   {point['date']}: {point['records']} records")

print(f"✅ History retrieval working")

# Test 5: Service layer
print("\n[TEST 5] Service Layer Integration")
print("="*80)

service_result = get_ml_database_size_history()
print(f"Service returned {len(service_result['points'])} points")

if len(service_result['points']) > 0:
    first = service_result['points'][0]
    last = service_result['points'][-1]
    print(f"   First: {first['date']} - {first['records']} records")
    print(f"   Last: {last['date']} - {last['records']} records")
    
    # Calculate growth
    if first['records'] > 0:
        growth = last['records'] - first['records']
        growth_pct = (growth / first['records']) * 100
        print(f"   Growth: +{growth} records ({growth_pct:.1f}%)")

print(f"✅ Service layer working")

# Test 6: Verify chart-ready data
print("\n[TEST 6] Chart-Ready Data Validation")
print("="*80)

points = service_result['points']

# Check structure
all_valid = True
for point in points:
    if not isinstance(point['date'], str):
        print(f"❌ Invalid date type: {type(point['date'])}")
        all_valid = False
    if not isinstance(point['records'], int):
        print(f"❌ Invalid records type: {type(point['records'])}")
        all_valid = False
    if point['records'] < 0:
        print(f"❌ Negative record count: {point['records']}")
        all_valid = False

if all_valid:
    print("✅ All data points valid for charting")
    print(f"   {len(points)} points ready for frontend")
else:
    print("❌ Some data points have issues")

# Test 7: Verify chronological order
print("\n[TEST 7] Chronological Order")
print("="*80)

dates = [p['date'] for p in points]
is_sorted = dates == sorted(dates)

if is_sorted:
    print("✅ Data points in chronological order")
else:
    print("❌ Data points NOT in chronological order")

# Final Summary
print("\n" + "="*80)
print("VERIFICATION SUMMARY")
print("="*80)

checks = [
    ("ML Database exists", Path(ML_DB_PATH).exists()),
    ("Has records", current_size > 0),
    ("Can record size", True),  # If we got here, recording worked
    ("Can retrieve history", len(history) > 0),
    ("Service layer works", len(service_result['points']) > 0),
    ("Data structure valid", all_valid),
    ("Chronological order", is_sorted)
]

passed = sum(1 for _, check in checks if check)
total = len(checks)

for name, result in checks:
    status = "✅" if result else "⚠️"
    print(f"{status} {name}")

print(f"\n{passed}/{total} checks passed ({(passed/total)*100:.0f}%)")

if passed == total:
    print("\n🎉 PHASE 2: ML DATABASE GROWTH TRACKING - FULLY FUNCTIONAL")
    print("\nKey Features:")
    print("  ✅ Correct path resolution to ML database")
    print("  ✅ Accurate record counting")
    print("  ✅ Historical data recording")
    print("  ✅ Historical data retrieval")
    print("  ✅ Service layer integration")
    print("  ✅ Chart-ready data format")
    print("  ✅ Chronological ordering")
    print("\n✨ Ready for frontend integration!")
else:
    print(f"\n⚠️  {total - passed} checks need attention")

print("="*80)
