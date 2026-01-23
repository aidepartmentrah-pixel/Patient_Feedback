"""
TEST PHASE 1: API INTEGRATION TEST
====================================
Test the /api/settings/training/progress endpoint with actual backend server.

Prerequisites:
- Backend server must be running
- Run this after starting uvicorn
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
print("PHASE 1: API INTEGRATION TEST")
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

def test_progress_endpoint_exists():
    """Test that GET /progress endpoint exists."""
    response = requests.get(f"{BASE_URL}/api/settings/training/progress")
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    print(f"   ✓ Endpoint exists and returns 200")


def test_progress_not_running():
    """Test progress response when not running."""
    response = requests.get(f"{BASE_URL}/api/settings/training/progress")
    data = response.json()
    
    # Should show not running (assuming no training active)
    assert "is_running" in data, "Response missing is_running field"
    assert isinstance(data["is_running"], bool), "is_running should be boolean"
    
    required_fields = [
        "is_running", "run_id", "current_model", "current_step",
        "total_steps", "progress_percentage", "elapsed_seconds",
        "estimated_remaining_seconds", "last_completed"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing field: {field}"
    
    print(f"   ✓ Response structure valid")
    print(f"   ✓ is_running: {data['is_running']}")
    print(f"   ✓ All required fields present")


def test_trigger_training_and_monitor():
    """Test starting training and monitoring progress."""
    # Start training
    print("\n   Starting training...")
    response = requests.post(f"{BASE_URL}/api/settings/training/run")
    assert response.status_code == 200, f"Failed to start training: {response.status_code}"
    
    result = response.json()
    assert "run_id" in result, "Missing run_id in response"
    assert "status" in result, "Missing status in response"
    assert result["status"] == "started", f"Expected 'started', got {result['status']}"
    
    run_id = result["run_id"]
    print(f"   ✓ Training started: {run_id}")
    
    # Poll progress
    print(f"   Monitoring progress...")
    poll_count = 0
    max_polls = 100  # Max ~3 minutes (100 * 2s)
    last_step = 0
    
    while poll_count < max_polls:
        time.sleep(2)  # Poll every 2 seconds
        poll_count += 1
        
        response = requests.get(f"{BASE_URL}/api/settings/training/progress")
        progress = response.json()
        
        if not progress["is_running"]:
            print(f"   ✓ Training completed after {poll_count * 2} seconds")
            break
        
        # Show progress updates
        current_step = progress["current_step"]
        if current_step > last_step:
            print(f"   [{progress['progress_percentage']}%] Step {current_step}/{progress['total_steps']}: {progress['current_model']}")
            print(f"      Elapsed: {progress['elapsed_seconds']}s, Remaining: ~{progress['estimated_remaining_seconds']}s")
            last_step = current_step
    
    # Verify completion
    final_response = requests.get(f"{BASE_URL}/api/settings/training/progress")
    final_progress = final_response.json()
    assert final_progress["is_running"] == False, "Training should have finished"
    
    print(f"   ✓ Training completed successfully")


def test_status_after_training():
    """Test status endpoint shows updated models."""
    response = requests.get(f"{BASE_URL}/api/settings/training/status")
    assert response.status_code == 200, f"Status endpoint failed: {response.status_code}"
    
    data = response.json()
    assert "models" in data, "Missing models field"
    
    models = data["models"]
    model_count = len(models)
    
    print(f"   ✓ Status shows {model_count} trained models")
    
    # Should have 18 models (not the mock 2)
    assert model_count >= 18, f"Expected at least 18 models, got {model_count}"
    
    print(f"   ✓ All models trained successfully")


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
test_api("GET /progress endpoint exists", test_progress_endpoint_exists)
test_api("Progress response when not running", test_progress_not_running)

# Ask user if they want to run the full training test
print("\n" + "="*80)
print("⚠️  FULL TRAINING TEST")
print("="*80)
print("The next test will trigger actual model training (~2-3 minutes).")
print("This will train all 18 models and monitor progress in real-time.")
response = input("\nProceed with full training test? (y/N): ").strip().lower()

if response == 'y':
    test_api("Trigger training and monitor progress", test_trigger_training_and_monitor)
    test_api("Status shows updated models", test_status_after_training)
else:
    print("⏭️  Skipping full training test")
    total_tests += 2
    print(f"\n   Note: 2 tests skipped (manual intervention required)")

# Print summary
print("\n" + "="*80)
print("API INTEGRATION TEST SUMMARY")
print("="*80)
print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")

if tests_failed == 0:
    print(f"\n🎉 ALL API TESTS PASSED!")
    if response != 'y':
        print("   Note: Run with full training test for complete verification")
else:
    print(f"\n⚠️  {tests_failed} test(s) failed")

print("="*80)
