"""
TEST PHASE 2: ML DATABASE GROWTH TRACKING
==========================================
Comprehensive test suite for ML database size tracking and history.

Tests:
1. ML database path resolution
2. Current ML DB size retrieval
3. DB size history recording
4. DB size history retrieval
5. Chart data structure
6. Integration with training pipeline
"""

import sys
import os
from pathlib import Path
from datetime import date, timedelta

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.api.db_layer.training_db import (
    get_current_ml_db_size,
    record_ml_db_size,
    get_ml_db_size_history,
    ML_DB_PATH,
    TRAINING_DB_PATH
)
from backend.api.services.training_service import get_ml_database_size_history

print("\n" + "="*80)
print("PHASE 2: ML DATABASE GROWTH TRACKING TEST SUITE")
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

@test_case("ML database path resolution")
def test_ml_db_path():
    """Test that ML database path is correctly resolved."""
    assert os.path.exists(ML_DB_PATH), f"ML database not found at: {ML_DB_PATH}"
    
    # Verify it's the correct file
    assert "patient_feedback_ml.db" in ML_DB_PATH, "Path should contain patient_feedback_ml.db"
    assert "models_directory" in ML_DB_PATH, "Path should contain models_directory"
    
    # Verify it's not in backend folder (common error)
    assert "backend\\models_directory" not in ML_DB_PATH and "backend/models_directory" not in ML_DB_PATH, \
        "Path should not be in backend folder"
    
    print(f"   ✓ ML DB path: {ML_DB_PATH}")
    print(f"   ✓ Database exists: {os.path.exists(ML_DB_PATH)}")


@test_case("Get current ML database size")
def test_get_current_size():
    """Test retrieving current ML database size."""
    size = get_current_ml_db_size()
    
    assert isinstance(size, int), f"Size should be integer, got {type(size)}"
    assert size >= 0, f"Size should be non-negative, got {size}"
    
    # Should have records if database has been used
    if os.path.exists(ML_DB_PATH):
        print(f"   ✓ Current ML DB size: {size} records")
        
        # If size is 0, warn but don't fail (might be fresh install)
        if size == 0:
            print(f"   ⚠️  Warning: Database exists but has 0 records")
            print(f"      This is OK for fresh install")
    else:
        print(f"   ⚠️  Warning: ML database doesn't exist yet")
        assert size == 0, "Size should be 0 if database doesn't exist"


@test_case("ML database table structure")
def test_ml_db_table():
    """Test that ML database has correct table structure."""
    if not os.path.exists(ML_DB_PATH):
        print("   ⏭️  Skipping - ML database doesn't exist yet")
        return
    
    import sqlite3
    conn = sqlite3.connect(ML_DB_PATH)
    cursor = conn.cursor()
    
    # Check table exists
    cursor.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name='patient_feedback_encoded'
    """)
    table = cursor.fetchone()
    assert table is not None, "Table 'patient_feedback_encoded' should exist"
    
    # Get column info
    cursor.execute("PRAGMA table_info(patient_feedback_encoded)")
    columns = [row[1] for row in cursor.fetchall()]
    
    # Verify key columns exist
    required_columns = ['id', 'feedback_received_date', 'complaint_text']
    for col in required_columns:
        assert col in columns, f"Column '{col}' should exist in table"
    
    conn.close()
    
    print(f"   ✓ Table exists with {len(columns)} columns")
    print(f"   ✓ All required columns present")


@test_case("Record ML database size - Single entry")
def test_record_size_single():
    """Test recording a single ML database size entry."""
    test_date = "2026-01-15"
    test_count = 100
    
    record_ml_db_size(test_count, test_date)
    
    # Verify it was recorded
    import sqlite3
    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT record_count FROM ml_db_size_history
        WHERE record_date = ?
    """, (test_date,))
    
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None, f"Record not found for date {test_date}"
    assert result[0] == test_count, f"Expected {test_count}, got {result[0]}"
    
    print(f"   ✓ Recorded {test_count} records for {test_date}")


@test_case("Record ML database size - Multiple entries")
def test_record_size_multiple():
    """Test recording multiple ML database size entries."""
    test_data = [
        ("2026-01-16", 150),
        ("2026-01-17", 200),
        ("2026-01-18", 275),
        ("2026-01-19", 350),
        ("2026-01-20", 425),
    ]
    
    for test_date, test_count in test_data:
        record_ml_db_size(test_count, test_date)
    
    # Verify all were recorded
    import sqlite3
    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()
    
    for test_date, expected_count in test_data:
        cursor.execute("""
            SELECT record_count FROM ml_db_size_history
            WHERE record_date = ?
        """, (test_date,))
        
        result = cursor.fetchone()
        assert result is not None, f"Record not found for {test_date}"
        assert result[0] == expected_count, f"Expected {expected_count}, got {result[0]}"
    
    conn.close()
    
    print(f"   ✓ Recorded {len(test_data)} entries successfully")


@test_case("Record ML database size - Replace existing")
def test_record_size_replace():
    """Test that recording replaces existing entry for same date."""
    test_date = "2026-01-21"
    
    # Record initial value
    record_ml_db_size(500, test_date)
    
    # Record new value for same date
    record_ml_db_size(608, test_date)
    
    # Verify only latest value exists
    import sqlite3
    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*), record_count FROM ml_db_size_history
        WHERE record_date = ?
        GROUP BY record_date
    """, (test_date,))
    
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None, "Record should exist"
    count, value = result
    assert count == 1, f"Should have only 1 record for date, got {count}"
    assert value == 608, f"Should have latest value 608, got {value}"
    
    print(f"   ✓ Successfully replaced old value with new value")


@test_case("Get ML database size history")
def test_get_size_history():
    """Test retrieving ML database size history."""
    history = get_ml_db_size_history(days=90)
    
    assert isinstance(history, list), "History should be a list"
    
    if len(history) > 0:
        # Check structure of first entry
        entry = history[0]
        assert "date" in entry, "Entry should have 'date' field"
        assert "records" in entry, "Entry should have 'records' field"
        
        # Verify chronological order (oldest first)
        if len(history) > 1:
            dates = [entry["date"] for entry in history]
            assert dates == sorted(dates), "History should be in chronological order"
        
        print(f"   ✓ Retrieved {len(history)} history entries")
        print(f"   ✓ Date range: {history[0]['date']} to {history[-1]['date']}")
    else:
        print(f"   ⚠️  No history entries found (OK for fresh install)")


@test_case("ML database size history - Data structure")
def test_history_structure():
    """Test the structure of history data for charting."""
    # Create test data
    today = date.today()
    for i in range(7):
        test_date = (today - timedelta(days=6-i)).isoformat()
        test_count = 100 + (i * 50)
        record_ml_db_size(test_count, test_date)
    
    # Get history
    history = get_ml_db_size_history(days=7)
    
    assert len(history) >= 7, f"Should have at least 7 entries, got {len(history)}"
    
    # Verify each entry has correct structure
    for entry in history:
        assert isinstance(entry["date"], str), "Date should be string"
        assert isinstance(entry["records"], int), "Records should be integer"
        assert entry["records"] >= 0, "Records should be non-negative"
    
    # Verify increasing trend in our test data
    last_7 = history[-7:]
    for i in range(1, len(last_7)):
        assert last_7[i]["records"] >= last_7[i-1]["records"], \
            "Test data should show increasing trend"
    
    print(f"   ✓ Data structure valid for charting")
    print(f"   ✓ Sample data: {last_7[0]} to {last_7[-1]}")


@test_case("Service layer - get_ml_database_size_history")
def test_service_layer():
    """Test service layer wrapper function."""
    result = get_ml_database_size_history()
    
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "points" in result, "Result should have 'points' key"
    assert isinstance(result["points"], list), "Points should be a list"
    
    if len(result["points"]) > 0:
        point = result["points"][0]
        assert "date" in point, "Point should have 'date' field"
        assert "records" in point, "Point should have 'records' field"
        
        print(f"   ✓ Service layer returns {len(result['points'])} points")
    else:
        print(f"   ⚠️  No points returned (OK for fresh install)")


@test_case("Record current ML database size")
def test_record_current_size():
    """Test recording actual current ML database size."""
    current_size = get_current_ml_db_size()
    today = date.today().isoformat()
    
    record_ml_db_size(current_size, today)
    
    # Verify it was recorded
    history = get_ml_db_size_history(days=1)
    
    today_entry = [e for e in history if e["date"] == today]
    assert len(today_entry) > 0, "Today's entry should exist"
    assert today_entry[0]["records"] == current_size, \
        f"Recorded size {today_entry[0]['records']} should match current {current_size}"
    
    print(f"   ✓ Recorded current size: {current_size} records")


@test_case("Negative record count validation")
def test_negative_count():
    """Test that negative counts are rejected."""
    test_date = "2026-01-22"
    
    # This should not raise an error, but should skip recording
    record_ml_db_size(-1, test_date)
    
    # Verify it was NOT recorded
    import sqlite3
    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT record_count FROM ml_db_size_history
        WHERE record_date = ? AND record_count < 0
    """, (test_date,))
    
    result = cursor.fetchone()
    conn.close()
    
    assert result is None, "Negative count should not be recorded"
    
    print(f"   ✓ Negative count properly rejected")


@test_case("History limit parameter")
def test_history_limit():
    """Test that history respects the days parameter."""
    # Get different limits
    history_7 = get_ml_db_size_history(days=7)
    history_30 = get_ml_db_size_history(days=30)
    history_90 = get_ml_db_size_history(days=90)
    
    # Verify limits are respected
    assert len(history_7) <= 7, f"7-day history should have ≤7 entries, got {len(history_7)}"
    assert len(history_30) <= 30, f"30-day history should have ≤30 entries, got {len(history_30)}"
    assert len(history_90) <= 90, f"90-day history should have ≤90 entries, got {len(history_90)}"
    
    # Verify ordering (30-day should include 7-day data)
    assert len(history_30) >= len(history_7), "30-day should have ≥ 7-day entries"
    
    print(f"   ✓ History limits respected: 7={len(history_7)}, 30={len(history_30)}, 90={len(history_90)}")


@test_case("Zero record count handling")
def test_zero_count():
    """Test that zero count is valid and recorded."""
    test_date = "2026-01-23"
    
    record_ml_db_size(0, test_date)
    
    # Verify it was recorded
    import sqlite3
    conn = sqlite3.connect(TRAINING_DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT record_count FROM ml_db_size_history
        WHERE record_date = ?
    """, (test_date,))
    
    result = cursor.fetchone()
    conn.close()
    
    assert result is not None, "Zero count should be recorded"
    assert result[0] == 0, "Count should be exactly 0"
    
    print(f"   ✓ Zero count properly recorded")


# ==================== RUN ALL TESTS ====================

print("\n" + "="*80)
print("RUNNING TEST SUITE")
print("="*80)

# Run all tests
test_ml_db_path()
test_get_current_size()
test_ml_db_table()
test_record_size_single()
test_record_size_multiple()
test_record_size_replace()
test_get_size_history()
test_history_structure()
test_service_layer()
test_record_current_size()
test_negative_count()
test_history_limit()
test_zero_count()

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
    print("\n🎉 ALL TESTS PASSED! Phase 2 implementation is complete and verified.")
    sys.exit(0)
else:
    print(f"\n⚠️ {tests_failed} test(s) failed. Please review and fix.")
    sys.exit(1)
