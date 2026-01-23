"""
TEST PHASE 1: PROGRESS TRACKING
================================
Comprehensive test suite for real-time training progress tracking.

Tests:
1. Progress state initialization
2. Progress updates during training
3. GET /progress endpoint
4. Progress calculations (percentage, time estimates)
5. Progress reset after training
"""

import sys
import time
from pathlib import Path
import threading

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.api.services.training_service import (
    initialize_training_progress,
    update_training_progress,
    get_training_progress,
    reset_training_progress,
    is_training_running
)

print("\n" + "="*80)
print("PHASE 1: PROGRESS TRACKING TEST SUITE")
print("="*80)

# Test counters
tests_passed = 0
tests_failed = 0
total_tests = 0


def test_case(name: str):
    """Decorator for test cases."""
    def decorator(func):
        def wrapper():
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
        return wrapper
    return decorator


# ==================== TEST SUITE ====================

@test_case("Initialize progress tracking")
def test_initialize_progress():
    """Test progress initialization."""
    run_id = "TEST_2026_01_21_1400"
    initialize_training_progress(run_id, total_steps=18)
    
    progress = get_training_progress()
    
    assert progress["is_running"] == True, "is_running should be True"
    assert progress["run_id"] == run_id, f"run_id mismatch: {progress['run_id']}"
    assert progress["total_steps"] == 18, f"total_steps should be 18: {progress['total_steps']}"
    assert progress["current_step"] == 0, f"current_step should be 0: {progress['current_step']}"
    assert progress["progress_percentage"] == 0, f"progress_percentage should be 0: {progress['progress_percentage']}"
    print(f"   ✓ Progress initialized: {progress['run_id']}")


@test_case("Update progress - First step")
def test_update_progress_first_step():
    """Test first progress update."""
    update_training_progress("Domain_Model", 1)
    
    progress = get_training_progress()
    
    assert progress["current_model"] == "Domain_Model", f"current_model mismatch: {progress['current_model']}"
    assert progress["current_step"] == 1, f"current_step should be 1: {progress['current_step']}"
    assert progress["progress_percentage"] > 0, "progress_percentage should be > 0"
    print(f"   ✓ Step 1: {progress['current_model']} ({progress['progress_percentage']}%)")


@test_case("Update progress - Multiple steps")
def test_update_progress_multiple_steps():
    """Test multiple progress updates."""
    steps = [
        ("Category_Domain1", 2),
        ("Category_Domain2", 3),
        ("Category_Domain3", 4),
        ("Subcategory_Cat1", 5)
    ]
    
    for model_name, step in steps:
        update_training_progress(model_name, step)
        progress = get_training_progress()
        
        assert progress["current_model"] == model_name, f"Model mismatch at step {step}"
        assert progress["current_step"] == step, f"Step mismatch: {progress['current_step']} != {step}"
        
        expected_percentage = int((step / 18) * 100)
        assert progress["progress_percentage"] == expected_percentage, \
            f"Percentage mismatch: {progress['progress_percentage']} != {expected_percentage}"
        
        print(f"   ✓ Step {step}/18: {model_name} ({progress['progress_percentage']}%)")


@test_case("Progress percentage calculation")
def test_progress_percentage():
    """Test progress percentage at various steps."""
    test_cases = [
        (1, 18, 5),    # Step 1 of 18 = 5%
        (5, 18, 27),   # Step 5 of 18 = 27%
        (9, 18, 50),   # Step 9 of 18 = 50%
        (18, 18, 100), # Step 18 of 18 = 100%
    ]
    
    for step, total, expected_pct in test_cases:
        update_training_progress(f"Model_{step}", step)
        progress = get_training_progress()
        
        assert progress["progress_percentage"] == expected_pct, \
            f"Step {step}/{total}: Expected {expected_pct}%, got {progress['progress_percentage']}%"
        
        print(f"   ✓ Step {step}/{total} = {expected_pct}%")


@test_case("Time estimation - Elapsed time")
def test_elapsed_time():
    """Test elapsed time calculation."""
    # Re-initialize with known start time
    initialize_training_progress("TEST_TIME_2026_01_21_1400", total_steps=18)
    
    # Wait a bit
    time.sleep(2)
    
    # Update progress
    update_training_progress("Test_Model", 1)
    
    progress = get_training_progress()
    elapsed = progress["elapsed_seconds"]
    
    assert elapsed >= 2, f"Elapsed time should be >= 2 seconds: {elapsed}"
    assert elapsed < 5, f"Elapsed time should be < 5 seconds: {elapsed}"
    
    print(f"   ✓ Elapsed time: {elapsed} seconds")


@test_case("Time estimation - Remaining time")
def test_remaining_time_estimation():
    """Test remaining time estimation."""
    # Initialize fresh
    initialize_training_progress("TEST_ESTIMATE_2026_01_21_1400", total_steps=18)
    
    # Simulate 3 steps taking 2 seconds
    time.sleep(2)
    update_training_progress("Model_3", 3)
    
    progress = get_training_progress()
    elapsed = progress["elapsed_seconds"]
    estimated_remaining = progress["estimated_remaining_seconds"]
    
    # With 3 steps done in ~2 seconds, avg is ~0.67s per step
    # Remaining 15 steps should take ~10 seconds
    assert estimated_remaining > 5, f"Estimated remaining should be > 5s: {estimated_remaining}"
    assert estimated_remaining < 15, f"Estimated remaining should be < 15s: {estimated_remaining}"
    
    print(f"   ✓ Elapsed: {elapsed}s, Estimated remaining: {estimated_remaining}s")


@test_case("Last completed model tracking")
def test_last_completed():
    """Test last_completed field updates."""
    initialize_training_progress("TEST_COMPLETED_2026_01_21_1400", total_steps=18)
    
    # First model - no previous
    update_training_progress("Model_A", 1)
    progress = get_training_progress()
    assert progress["last_completed"] == "Initializing...", \
        f"First step last_completed should be 'Initializing...': {progress['last_completed']}"
    
    # Second model - should show Model_A
    update_training_progress("Model_B", 2)
    progress = get_training_progress()
    assert progress["last_completed"] == "Model_A", \
        f"Second step last_completed should be 'Model_A': {progress['last_completed']}"
    
    # Third model - should show Model_B
    update_training_progress("Model_C", 3)
    progress = get_training_progress()
    assert progress["last_completed"] == "Model_B", \
        f"Third step last_completed should be 'Model_B': {progress['last_completed']}"
    
    print(f"   ✓ Last completed tracking works correctly")


@test_case("Progress when not running")
def test_progress_not_running():
    """Test progress response when training is not running."""
    # Reset progress
    reset_training_progress()
    
    progress = get_training_progress()
    
    assert progress["is_running"] == False, "is_running should be False"
    assert progress["run_id"] is None, "run_id should be None"
    assert progress["current_model"] is None, "current_model should be None"
    assert progress["current_step"] == 0, "current_step should be 0"
    assert progress["progress_percentage"] == 0, "progress_percentage should be 0"
    assert progress["elapsed_seconds"] == 0, "elapsed_seconds should be 0"
    assert progress["estimated_remaining_seconds"] == 0, "estimated_remaining_seconds should be 0"
    
    print(f"   ✓ Progress correctly shows not running")


@test_case("Reset progress state")
def test_reset_progress():
    """Test progress reset functionality."""
    # Set up some progress
    initialize_training_progress("TEST_RESET_2026_01_21_1400", total_steps=18)
    update_training_progress("Some_Model", 5)
    
    # Verify it's running
    progress = get_training_progress()
    assert progress["is_running"] == True, "Should be running before reset"
    assert progress["current_step"] == 5, "Should have progress before reset"
    
    # Reset
    reset_training_progress()
    
    # Verify reset
    progress = get_training_progress()
    assert progress["is_running"] == False, "Should not be running after reset"
    assert progress["current_step"] == 0, "Step should be 0 after reset"
    assert progress["current_model"] is None, "Model should be None after reset"
    
    print(f"   ✓ Progress reset successful")


@test_case("Progress at 100% completion")
def test_full_completion():
    """Test progress at 100% completion."""
    initialize_training_progress("TEST_100_2026_01_21_1400", total_steps=18)
    
    # Update to final step
    update_training_progress("Final_Model", 18)
    
    progress = get_training_progress()
    
    assert progress["current_step"] == 18, "Should be at step 18"
    assert progress["progress_percentage"] == 100, f"Should be 100%: {progress['progress_percentage']}"
    assert progress["estimated_remaining_seconds"] == 0, "Should have 0 seconds remaining"
    
    print(f"   ✓ 100% completion handled correctly")


@test_case("Progress response structure")
def test_progress_response_structure():
    """Test that progress response has all required fields."""
    initialize_training_progress("TEST_STRUCT_2026_01_21_1400", total_steps=18)
    update_training_progress("Test_Model", 1)
    
    progress = get_training_progress()
    
    required_fields = [
        "is_running",
        "run_id",
        "current_model",
        "current_step",
        "total_steps",
        "progress_percentage",
        "elapsed_seconds",
        "estimated_remaining_seconds",
        "last_completed"
    ]
    
    for field in required_fields:
        assert field in progress, f"Missing required field: {field}"
    
    # Type checks
    assert isinstance(progress["is_running"], bool), "is_running should be bool"
    assert isinstance(progress["current_step"], int), "current_step should be int"
    assert isinstance(progress["total_steps"], int), "total_steps should be int"
    assert isinstance(progress["progress_percentage"], int), "progress_percentage should be int"
    assert isinstance(progress["elapsed_seconds"], int), "elapsed_seconds should be int"
    assert isinstance(progress["estimated_remaining_seconds"], int), "estimated_remaining_seconds should be int"
    
    print(f"   ✓ Response structure valid with all required fields")


# ==================== RUN ALL TESTS ====================

print("\n" + "="*80)
print("RUNNING TEST SUITE")
print("="*80)

# Run all tests
test_initialize_progress()
test_update_progress_first_step()
test_update_progress_multiple_steps()
test_progress_percentage()
test_elapsed_time()
test_remaining_time_estimation()
test_last_completed()
test_progress_not_running()
test_reset_progress()
test_full_completion()
test_progress_response_structure()

# Print summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)
print(f"Total Tests: {total_tests}")
print(f"✅ Passed: {tests_passed}")
print(f"❌ Failed: {tests_failed}")
print(f"Success Rate: {(tests_passed/total_tests)*100:.1f}%")
print("="*80)

if tests_failed == 0:
    print("\n🎉 ALL TESTS PASSED! Phase 1 implementation is complete and verified.")
    sys.exit(0)
else:
    print(f"\n⚠️ {tests_failed} test(s) failed. Please review and fix.")
    sys.exit(1)
