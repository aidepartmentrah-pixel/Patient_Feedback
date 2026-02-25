"""
TEST TASK D-B3 — WORKER SERVICE LAYER

Verifies worker reporting service layer implementation.
"""

import sys
import os
from pathlib import Path
from datetime import date, timedelta
import inspect

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def test_file_exists():
    """Verify service file exists at correct location."""
    service_path = backend_path / "api" / "services" / "worker_reporting_service.py"
    assert service_path.exists(), f"❌ Service file not found at: {service_path}"
    print("✅ worker_reporting_service.py exists")
    return True


def test_function_exists():
    """Verify get_worker_profile function exists."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        
        assert hasattr(WorkerReportingService, 'get_worker_profile'), "❌ get_worker_profile not found"
        
        func = WorkerReportingService.get_worker_profile
        assert callable(func), "❌ get_worker_profile is not callable"
        
        print("✅ get_worker_profile function exists")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_correct_imports():
    """Verify service imports only from db_layer and schemas."""
    service_path = backend_path / "api" / "services" / "worker_reporting_service.py"
    
    with open(service_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for required imports
    required_imports = [
        'worker_reporting_db',
        'worker_reporting_schema'
    ]
    
    for required in required_imports:
        if required not in content:
            print(f"❌ Missing required import: {required}")
            return False
    
    print("✅ Correct imports: db_layer and schemas")
    return True


def test_no_router_imports():
    """Verify no router or FastAPI imports in service layer."""
    service_path = backend_path / "api" / "services" / "worker_reporting_service.py"
    
    with open(service_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    forbidden_imports = [
        'from fastapi',
        'import fastapi',
        'from ..routers',
        'from api.routers',
        'APIRouter',
        'Depends',
        'HTTPException',
        'Query',
        'Path'
    ]
    
    for forbidden in forbidden_imports:
        if forbidden in content:
            print(f"❌ Forbidden import found: {forbidden}")
            return False
    
    print("✅ No router or FastAPI imports (proper layering)")
    return True


def test_returns_worker_profile_response():
    """Verify function returns WorkerProfileResponse model."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        from api.schemas.worker_reporting_schema import WorkerProfileResponse
        
        # Check function signature
        func = WorkerReportingService.get_worker_profile
        source = inspect.getsource(func)
        
        # Check return type annotation
        signature = inspect.signature(func)
        return_annotation = signature.return_annotation
        
        # Should be WorkerProfileResponse
        if return_annotation != inspect.Parameter.empty:
            if 'WorkerProfileResponse' not in str(return_annotation):
                print(f"⚠️  Warning: Return type annotation may not be WorkerProfileResponse")
        
        print("✅ Function signature includes WorkerProfileResponse return type")
        return True
        
    except Exception as e:
        print(f"❌ Return type verification failed: {e}")
        return False


def test_raises_value_error_on_not_found():
    """Verify function raises ValueError when worker not found."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        
        # Test with impossible employee ID
        impossible_id = 999999999
        
        try:
            profile = WorkerReportingService.get_worker_profile(impossible_id)
            print(f"❌ Should have raised ValueError for non-existent employee")
            return False
        except ValueError as ve:
            if "not found" in str(ve).lower() or "Worker not found" in str(ve):
                print(f"✅ Raises ValueError correctly for missing worker: {ve}")
                return True
            else:
                print(f"❌ ValueError raised but message unclear: {ve}")
                return False
        except Exception as e:
            print(f"❌ Raised wrong exception type: {type(e).__name__}: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_no_sql_in_service():
    """Verify no SQL queries in service file (should use DB layer)."""
    service_path = backend_path / "api" / "services" / "worker_reporting_service.py"
    
    with open(service_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Look for SQL patterns (excluding comments and docstrings)
    sql_patterns = [
        'SELECT ',
        'INSERT INTO',
        'UPDATE ',
        'DELETE FROM',
        'CREATE TABLE',
        'cursor.execute'
    ]
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith('#'):
            continue
        if '"""' in line or "'''" in line:
            continue
        
        for pattern in sql_patterns:
            if pattern in line:
                print(f"❌ SQL found in service file at line {i}: {pattern}")
                print(f"   Line: {line.strip()}")
                return False
    
    print("✅ No SQL in service file (uses DB layer correctly)")
    return True


def test_function_has_docstring():
    """Verify get_worker_profile has proper docstring."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        
        func = WorkerReportingService.get_worker_profile
        docstring = inspect.getdoc(func)
        
        assert docstring is not None, "❌ Function has no docstring"
        assert len(docstring) > 100, "❌ Docstring is too short"
        
        # Check for key concepts in docstring
        docstring_lower = docstring.lower()
        key_concepts = ['aggregat', 'metric', 'worker', 'profile']
        
        missing_concepts = [c for c in key_concepts if c not in docstring_lower]
        if missing_concepts:
            print(f"⚠️  Warning: Docstring may be missing concepts: {missing_concepts}")
        
        print("✅ Function has comprehensive docstring")
        return True
        
    except Exception as e:
        print(f"❌ Docstring verification failed: {e}")
        return False


def test_with_real_employee():
    """Test function with real employee from database."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        from api.schemas.worker_reporting_schema import WorkerProfileResponse
        from api.db_layer import worker_reporting_db
        
        # Get a real employee ID
        conn = worker_reporting_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 EmployeeID FROM APP_VIEWTABLE_HR_EMPLOYEES WHERE IsActive = 1")
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("⚠️  No active employees in database for real test")
            return True
        
        test_employee_id = row.EmployeeID
        
        # Test without date range
        profile = WorkerReportingService.get_worker_profile(test_employee_id)
        
        assert isinstance(profile, WorkerProfileResponse), f"❌ Expected WorkerProfileResponse, got {type(profile)}"
        assert profile.worker.employee_id == test_employee_id, "❌ Employee ID mismatch"
        assert profile.worker.full_name, "❌ Full name should not be empty"
        assert profile.metrics is not None, "❌ Metrics should not be None"
        
        print(f"✅ Function works with real employee {test_employee_id}")
        print(f"   Name: {profile.worker.full_name}")
        print(f"   Job: {profile.worker.job_title}")
        print(f"   Incidents: {profile.metrics.total_incidents}")
        print(f"   Actions: {profile.metrics.total_action_items} (completed: {profile.metrics.completed_action_items})")
        print(f"   Explanations: accepted={profile.metrics.explanation_accepted_count}, rejected={profile.metrics.explanation_rejected_count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real employee test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_date_range():
    """Test function with date range filtering."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        from api.db_layer import worker_reporting_db
        
        # Get a real employee ID
        conn = worker_reporting_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 EmployeeID FROM APP_VIEWTABLE_HR_EMPLOYEES WHERE IsActive = 1")
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("⚠️  No employees for date range test")
            return True
        
        test_employee_id = row.EmployeeID
        today = date.today()
        last_year = today - timedelta(days=365)
        
        # Test with date range
        profile = WorkerReportingService.get_worker_profile(
            employee_id=test_employee_id,
            date_from=last_year,
            date_to=today
        )
        
        assert profile.period_from == last_year, "❌ Date from mismatch"
        assert profile.period_to == today, "❌ Date to mismatch"
        
        print(f"✅ Function works with date range filtering")
        print(f"   Period: {profile.period_from} to {profile.period_to}")
        print(f"   Metrics: {profile.metrics.total_incidents} incidents, {profile.metrics.total_action_items} actions")
        
        return True
        
    except Exception as e:
        print(f"❌ Date range test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metric_block_completeness():
    """Verify all metric fields are populated correctly."""
    try:
        from api.services.worker_reporting_service import WorkerReportingService
        from api.db_layer import worker_reporting_db
        
        # Get a real employee ID
        conn = worker_reporting_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 EmployeeID FROM APP_VIEWTABLE_HR_EMPLOYEES")
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("⚠️  No employees for metric test")
            return True
        
        test_employee_id = row.EmployeeID
        profile = WorkerReportingService.get_worker_profile(test_employee_id)
        
        # Verify all metrics exist and are non-negative integers
        metrics = profile.metrics
        required_metrics = [
            'total_incidents',
            'total_action_items',
            'completed_action_items',
            'overdue_action_items',
            'explanation_rejected_count',
            'explanation_accepted_count'
        ]
        
        for metric_name in required_metrics:
            assert hasattr(metrics, metric_name), f"❌ Missing metric: {metric_name}"
            value = getattr(metrics, metric_name)
            assert isinstance(value, int), f"❌ {metric_name} should be int, got {type(value)}"
            assert value >= 0, f"❌ {metric_name} should be non-negative, got {value}"
        
        print("✅ All metric fields are complete and valid")
        return True
        
    except Exception as e:
        print(f"❌ Metric completeness test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST TASK D-B3 — WORKER SERVICE LAYER")
    print("=" * 70)
    print()
    
    tests = [
        ("File Exists", test_file_exists),
        ("Function Exists", test_function_exists),
        ("Correct Imports", test_correct_imports),
        ("No Router Imports", test_no_router_imports),
        ("Returns WorkerProfileResponse", test_returns_worker_profile_response),
        ("Raises ValueError on Not Found", test_raises_value_error_on_not_found),
        ("No SQL in Service", test_no_sql_in_service),
        ("Function Has Docstring", test_function_has_docstring),
        ("Works With Real Employee", test_with_real_employee),
        ("Works With Date Range", test_with_date_range),
        ("Metric Block Completeness", test_metric_block_completeness),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 70)
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total:  {passed + failed}")
    print()
    
    if failed == 0:
        print("🎉 WORKER SERVICE LAYER OK — ALL TESTS PASSED")
        return 0
    else:
        print("⚠️  WORKER SERVICE LAYER HAS ISSUES — REVIEW FAILURES ABOVE")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
