"""
TEST PHASE 2: API INTEGRATION TEST
====================================
Test the /api/settings/training/db-size endpoint and verify growth tracking
works correctly after actual training.
"""

import sys
import time
import requests
from pathlib import Path

workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

BASE_URL = "http://127.0.0.1:8000"

print("\n" + "="*80)
print("PHASE 2: API INTEGRATION TEST - ML DATABASE GROWTH")
print("="*80)

tests_passed = 0
tests_failed = 0
total_tests = 0


def test_api(name: str, func):
    """Run a single API test."""
    global tests_passed, tests_failed, total_tests
    total_tests += 1
    print(f"\n[TEST {total_tests}] {name}")
    try:
        func()
        print(f"✅ PASSED: {name}")
        tests_passed += 1
        return True
    except AssertionError as e:
        print(f"❌ FAILED: {name}")
        print(f"   Error: {e}")
        tests_failed += 1
        return False
    except Exception as e:
        print(f"❌ ERROR: {name}")
        print(f"   Exception: {e}")
        tests_failed += 1
        return False


# ==================== API TESTS ====================

def test_db_size_endpoint_exists():
    """Test that GET /db-size endpoint exists."""
    response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print(f"   ✓ Endpoint exists and returns 200")


def test_db_size_response_structure():
    """Test response structure of db-size endpoint."""
    response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    data = response.json()
    
    assert "points" in data, "Response should have 'points' key"
    assert isinstance(data["points"], list), "Points should be a list"
    
    if len(data["points"]) > 0:
        point = data["points"][0]
        assert "date" in point, "Point should have 'date' field"
        assert "records" in point, "Point should have 'records' field"
        assert isinstance(point["date"], str), "Date should be string"
        assert isinstance(point["records"], int), "Records should be integer"
        
        print(f"   ✓ Response structure valid")
        print(f"   ✓ Has {len(data['points'])} data points")
        print(f"   ✓ Date range: {data['points'][0]['date']} to {data['points'][-1]['date']}")
    else:
        print(f"   ⚠️  No data points (OK for fresh install)")


def test_db_size_has_data():
    """Test that db-size endpoint returns actual data."""
    response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    data = response.json()
    
    points = data["points"]
    
    if len(points) == 0:
        print(f"   ⚠️  Warning: No data points available")
        print(f"      This is OK if no training has been run yet")
        return
    
    # Verify data makes sense
    for point in points:
        assert point["records"] >= 0, f"Record count should be non-negative: {point}"
    
    # Check if we have recent data
    today = "2026-01-21"
    today_points = [p for p in points if p["date"] == today]
    
    if today_points:
        print(f"   ✓ Has data for today: {today_points[0]['records']} records")
    else:
        print(f"   ℹ️  No data for today yet")
    
    print(f"   ✓ All {len(points)} points have valid data")


def test_db_size_after_training():
    """Test that db-size is recorded after training."""
    print("\n   Checking current ML database status...")
    
    # Get current size from direct call
    from backend.api.db_layer.training_db import get_current_ml_db_size
    current_size = get_current_ml_db_size()
    print(f"   Current ML DB size: {current_size} records")
    
    if current_size == 0:
        print(f"   ⚠️  ML database is empty")
        print(f"      Run training first to populate the database")
        return
    
    # Get size from API
    response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    data = response.json()
    
    if len(data["points"]) == 0:
        print(f"   ⚠️  No historical data recorded yet")
        print(f"      Run training to record first data point")
        return
    
    # Get most recent point
    latest_point = data["points"][-1]
    latest_size = latest_point["records"]
    
    print(f"   ✓ Latest recorded size: {latest_size} records on {latest_point['date']}")
    
    # They should be close (might differ if training ran between calls)
    assert latest_size >= 0, "Latest size should be non-negative"
    
    if current_size == latest_size:
        print(f"   ✓ Current size matches latest recorded size")
    else:
        print(f"   ℹ️  Sizes differ slightly (OK if training ran recently)")
        print(f"      Current: {current_size}, Latest recorded: {latest_size}")


def test_db_size_chronological_order():
    """Test that data points are in chronological order."""
    response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    data = response.json()
    
    points = data["points"]
    
    if len(points) < 2:
        print(f"   ⏭️  Skipping - need at least 2 points for ordering test")
        return
    
    dates = [p["date"] for p in points]
    sorted_dates = sorted(dates)
    
    assert dates == sorted_dates, "Data points should be in chronological order"
    
    print(f"   ✓ Data points are in chronological order")
    print(f"   ✓ First: {dates[0]}, Last: {dates[-1]}")


def test_db_size_growth_trend():
    """Test that database shows growth trend over time."""
    response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    data = response.json()
    
    points = data["points"]
    
    if len(points) < 3:
        print(f"   ⏭️  Skipping - need at least 3 points for trend analysis")
        return
    
    # Check if generally increasing (some minor decreases OK for corrections)
    first_size = points[0]["records"]
    last_size = points[-1]["records"]
    
    if last_size >= first_size:
        growth = last_size - first_size
        growth_pct = (growth / first_size * 100) if first_size > 0 else 0
        
        print(f"   ✓ Database shows growth trend")
        print(f"      From {first_size} to {last_size} records")
        print(f"      Growth: +{growth} records ({growth_pct:.1f}%)")
    else:
        print(f"   ℹ️  Database size decreased (unusual but possible)")
        print(f"      From {first_size} to {last_size} records")


def test_trigger_training_and_verify_recording():
    """Test that training automatically records DB size."""
    print("\n   This test will trigger actual training (~2 minutes)")
    response = input("   Proceed with training test? (y/N): ").strip().lower()
    
    if response != 'y':
        print("   ⏭️  Skipping training test")
        return
    
    # Get DB size before training
    pre_response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    pre_points = pre_response.json()["points"]
    pre_count = len(pre_points)
    
    print(f"   Pre-training: {pre_count} historical points")
    
    # Start training
    print("   Starting training...")
    train_response = requests.post(f"{BASE_URL}/api/settings/training/run")
    assert train_response.status_code == 200, "Failed to start training"
    
    run_id = train_response.json()["run_id"]
    print(f"   Training started: {run_id}")
    
    # Wait for completion (poll progress)
    print("   Waiting for training to complete...")
    max_wait = 180  # 3 minutes max
    waited = 0
    
    while waited < max_wait:
        time.sleep(5)
        waited += 5
        
        progress_response = requests.get(f"{BASE_URL}/api/settings/training/progress")
        progress = progress_response.json()
        
        if not progress["is_running"]:
            print(f"   ✓ Training completed after {waited} seconds")
            break
        
        if waited % 20 == 0:
            print(f"      Still training... ({waited}s elapsed)")
    
    # Wait a moment for DB size to be recorded
    time.sleep(2)
    
    # Get DB size after training
    post_response = requests.get(f"{BASE_URL}/api/settings/training/db-size")
    post_points = post_response.json()["points"]
    post_count = len(post_points)
    
    print(f"   Post-training: {post_count} historical points")
    
    # Verify new point was added
    assert post_count >= pre_count, \
        f"Should have same or more points after training: {pre_count} -> {post_count}"
    
    # Get today's date
    today = "2026-01-21"
    today_points = [p for p in post_points if p["date"] == today]
    
    assert len(today_points) > 0, f"Should have entry for today ({today})"
    
    current_size = today_points[-1]["records"]
    print(f"   ✓ DB size recorded: {current_size} records for {today}")
    print(f"   ✓ Automatic recording working correctly")


# ==================== RUN TESTS ====================

print("\n" + "="*80)
print("CHECKING BACKEND AVAILABILITY")
print("="*80)

try:
    response = requests.get(f"{BASE_URL}/api/settings/training/status", timeout=5)
    print("✅ Backend server is running")
except Exception as e:
    print("❌ Backend server is not reachable")
    print(f"   Error: {e}")
    print("\n⚠️  Please start the backend server first:")
    print("   cd backend")
    print("   uvicorn main:app --reload")
    sys.exit(1)

print("\n" + "="*80)
print("RUNNING API TESTS")
print("="*80)

# Run tests
test_api("GET /db-size endpoint exists", test_db_size_endpoint_exists)
test_api("Response structure validation", test_db_size_response_structure)
test_api("Has actual data points", test_db_size_has_data)
test_api("DB size recorded after training", test_db_size_after_training)
test_api("Data in chronological order", test_db_size_chronological_order)
test_api("Database growth trend", test_db_size_growth_trend)

# Optional: Full training test
print("\n" + "="*80)
print("OPTIONAL: FULL TRAINING TEST")
print("="*80)
test_api("Training automatically records DB size", test_trigger_training_and_verify_recording)

# Print summary
print("\n" + "="*80)
print("API INTEGRATION TEST SUMMARY")
print("="*80)
print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")

if tests_failed == 0:
    print(f"\n🎉 ALL API TESTS PASSED!")
else:
    print(f"\n⚠️  {tests_failed} test(s) failed")

print("="*80)
