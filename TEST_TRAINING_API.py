"""
Test script for Training API endpoints
Run this to verify all 4 endpoints are working correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.api.services.training_service import (
    run_training_pipeline,
    get_training_status,
    get_training_history_list,
    get_ml_database_size_history
)
from backend.api.db_layer.training_db import record_ml_db_size

print("\n" + "="*80)
print("TRAINING API TEST")
print("="*80)

# Test 1: Record some ML DB size history (for testing)
print("\n[1] Recording ML DB size history...")
record_ml_db_size(153, "2026-01-01")
record_ml_db_size(191, "2026-01-02")
record_ml_db_size(248, "2026-01-03")
record_ml_db_size(312, "2026-01-04")
record_ml_db_size(371, "2026-01-05")
print("✓ Recorded 5 days of history")

# Test 2: Get status
print("\n[2] Testing GET /api/settings/training/status...")
status = get_training_status()
print(f"✓ Status: {status['status']}")
print(f"✓ Last run: {status['last_run']}")
print(f"✓ Models: {len(status['models'])}")

# Test 3: Get history
print("\n[3] Testing GET /api/settings/training/history...")
history = get_training_history_list()
print(f"✓ Retrieved {len(history['history'])} historical runs")

# Test 4: Get DB size history
print("\n[4] Testing GET /api/settings/training/db-size...")
db_size = get_ml_database_size_history()
print(f"✓ Retrieved {len(db_size['points'])} data points")
for point in db_size['points']:
    print(f"  {point['date']}: {point['records']} records")

# Test 5: Start training (don't wait for completion)
print("\n[5] Testing POST /api/settings/training/run...")
result = run_training_pipeline()
print(f"✓ Training started with run_id: {result['run_id']}")
print(f"✓ Status: {result['status']}")

print("\n" + "="*80)
print("✓✓✓ ALL TESTS PASSED ✓✓✓")
print("="*80 + "\n")
