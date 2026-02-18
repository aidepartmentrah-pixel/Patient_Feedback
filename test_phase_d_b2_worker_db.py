"""
TEST TASK D-B2 — WORKER DB AGGREGATION FUNCTIONS

Verifies worker reporting database layer implementation.
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
    """Verify DB layer file exists at correct location."""
    db_file_path = backend_path / "api" / "db_layer" / "worker_reporting_db.py"
    assert db_file_path.exists(), f"❌ DB file not found at: {db_file_path}"
    print("✅ worker_reporting_db.py exists")
    return True


def test_functions_exist():
    """Verify all required functions are defined."""
    try:
        from api.db_layer import worker_reporting_db
        
        required_functions = [
            'get_worker_identity',
            'count_worker_incidents',
            'count_worker_action_items',
            'count_worker_explanation_status'
        ]
        
        for func_name in required_functions:
            assert hasattr(worker_reporting_db, func_name), f"❌ Missing function: {func_name}"
            func = getattr(worker_reporting_db, func_name)
            assert callable(func), f"❌ {func_name} is not callable"
        
        print(f"✅ All {len(required_functions)} required functions exist")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_get_connection_pattern():
    """Verify get_connection() function exists and follows standard pattern."""
    try:
        from api.db_layer import worker_reporting_db
        
        assert hasattr(worker_reporting_db, 'get_connection'), "❌ get_connection() not found"
        assert callable(worker_reporting_db.get_connection), "❌ get_connection is not callable"
        
        # Verify it's a function, not imported from elsewhere
        source = inspect.getsource(worker_reporting_db.get_connection)
        assert 'pyodbc.connect' in source, "❌ get_connection doesn't use pyodbc.connect"
        assert 'SOCIALMEDIA' in source, "❌ get_connection doesn't use correct server"
        
        print("✅ get_connection() pattern is correct")
        return True
        
    except Exception as e:
        print(f"❌ get_connection verification failed: {e}")
        return False


def test_queries_are_select_only():
    """Verify all query functions use SELECT statements only (no INSERT/UPDATE/DELETE)."""
    try:
        from api.db_layer import worker_reporting_db
        
        functions_to_check = [
            'get_worker_identity',
            'count_worker_incidents',
            'count_worker_action_items',
            'count_worker_explanation_status'
        ]
        
        # Keywords that indicate write operations (not read-only)
        forbidden_keywords = ['INSERT INTO', 'UPDATE ', 'DELETE FROM', 'DROP ', 'TRUNCATE ', 'ALTER ', 'CREATE TABLE']
        
        for func_name in functions_to_check:
            func = getattr(worker_reporting_db, func_name)
            source = inspect.getsource(func)
            source_upper = source.upper()
            
            # Look for forbidden SQL patterns (not just keywords that might be in column names)
            for keyword in forbidden_keywords:
                if keyword in source_upper:
                    # Extract SQL portions only
                    lines = source.split('\n')
                    for line in lines:
                        line_upper = line.upper()
                        # Skip comments and docstrings
                        if line.strip().startswith('#'):
                            continue
                        if '"""' in line or "'''" in line:
                            continue
                        # Check if keyword appears in SQL statement context
                        if keyword in line_upper:
                            # Exclude if it's part of a column name like CreatedByUserID
                            if 'CREATE' in keyword and ('CREATEDAT' in line_upper or 'CREATEDBY' in line_upper):
                                continue
                            print(f"❌ {func_name} contains forbidden SQL operation: {keyword}")
                            return False
            
            # Verify SELECT is present
            if 'SELECT' not in source_upper:
                print(f"⚠️  Warning: {func_name} doesn't contain SELECT (might be placeholder)")
        
        print("✅ All query functions are read-only (SELECT only)")
        return True
        
    except Exception as e:
        print(f"❌ Query verification failed: {e}")
        return False


def test_no_service_imports():
    """Verify DB layer doesn't import from services layer."""
    db_file_path = backend_path / "api" / "db_layer" / "worker_reporting_db.py"
    
    with open(db_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    forbidden_imports = [
        'from ..services',
        'from api.services',
        'import services',
        'from services'
    ]
    
    for forbidden in forbidden_imports:
        if forbidden in content:
            print(f"❌ Forbidden service import found: {forbidden}")
            return False
    
    print("✅ No service layer imports found (proper layering)")
    return True


def test_no_pydantic_imports():
    """Verify DB layer doesn't import Pydantic models."""
    db_file_path = backend_path / "api" / "db_layer" / "worker_reporting_db.py"
    
    with open(db_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pydantic_imports = [
        'from pydantic',
        'import pydantic',
        'BaseModel'
    ]
    
    for pydantic_ref in pydantic_imports:
        if pydantic_ref in content:
            print(f"❌ Pydantic import found: {pydantic_ref}")
            return False
    
    print("✅ No Pydantic imports (returns dicts/scalars only)")
    return True


def test_return_types():
    """Verify functions return dicts or scalars, not Pydantic models."""
    try:
        from api.db_layer import worker_reporting_db
        
        # Check return type annotations
        func_return_types = {
            'get_worker_identity': 'Dict',  # Optional[Dict[str, Any]]
            'count_worker_incidents': 'int',
            'count_worker_action_items': 'Dict',  # Dict[str, int]
            'count_worker_explanation_status': 'Dict'  # Dict[int, int]
        }
        
        for func_name, expected_type in func_return_types.items():
            func = getattr(worker_reporting_db, func_name)
            source = inspect.getsource(func)
            
            # Check that BaseModel is not in return annotation
            if 'BaseModel' in source or '-> Worker' in source or '-> Employee' in source:
                print(f"❌ {func_name} appears to return a Pydantic model")
                return False
            
            # Verify expected return type is mentioned
            if expected_type not in source:
                print(f"⚠️  Warning: {func_name} may not have correct return type annotation")
        
        print("✅ All functions return dicts or scalars (not Pydantic models)")
        return True
        
    except Exception as e:
        print(f"❌ Return type verification failed: {e}")
        return False


def test_get_worker_identity_function():
    """Test get_worker_identity with live database query."""
    try:
        from api.db_layer import worker_reporting_db
        
        # First, get any employee ID from the database
        conn = worker_reporting_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 EmployeeID FROM APP_VIEWTABLE_HR_EMPLOYEES WHERE IsActive = 1")
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("⚠️  No active employees found in database (schema may be empty)")
            return True
        
        test_employee_id = row.EmployeeID
        
        # Test the function
        result = worker_reporting_db.get_worker_identity(test_employee_id)
        
        assert result is not None, "❌ get_worker_identity returned None for valid employee"
        assert isinstance(result, dict), f"❌ Expected dict, got {type(result)}"
        
        required_keys = [
            'employee_id', 'full_name', 'job_title', 'department_id',
            'section_id', 'administration_id', 'is_active'
        ]
        
        for key in required_keys:
            assert key in result, f"❌ Missing key in result: {key}"
        
        assert result['employee_id'] == test_employee_id, "❌ Employee ID mismatch"
        assert isinstance(result['full_name'], str), "❌ full_name should be string"
        
        print(f"✅ get_worker_identity works correctly (tested with employee {test_employee_id})")
        return True
        
    except Exception as e:
        print(f"❌ get_worker_identity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_count_functions_return_correct_types():
    """Test count functions return correct types (even if counts are zero)."""
    try:
        from api.db_layer import worker_reporting_db
        
        # Get any employee ID
        conn = worker_reporting_db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT TOP 1 EmployeeID FROM APP_VIEWTABLE_HR_EMPLOYEES")
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            print("⚠️  No employees found for count function tests")
            return True
        
        test_employee_id = row.EmployeeID
        today = date.today()
        last_year = today - timedelta(days=365)
        
        # Test count_worker_incidents
        incident_count = worker_reporting_db.count_worker_incidents(
            test_employee_id, last_year, today
        )
        assert isinstance(incident_count, int), f"❌ count_worker_incidents should return int, got {type(incident_count)}"
        assert incident_count >= 0, "❌ Incident count should be non-negative"
        
        # Test count_worker_action_items
        action_counts = worker_reporting_db.count_worker_action_items(
            test_employee_id, last_year, today
        )
        assert isinstance(action_counts, dict), f"❌ count_worker_action_items should return dict, got {type(action_counts)}"
        assert 'total' in action_counts, "❌ action_counts missing 'total' key"
        assert 'completed' in action_counts, "❌ action_counts missing 'completed' key"
        assert 'overdue' in action_counts, "❌ action_counts missing 'overdue' key"
        
        # Test count_worker_explanation_status
        explanation_counts = worker_reporting_db.count_worker_explanation_status(
            test_employee_id, None, last_year, today
        )
        assert isinstance(explanation_counts, dict), f"❌ count_worker_explanation_status should return dict, got {type(explanation_counts)}"
        
        print("✅ All count functions return correct types")
        print(f"   - Incidents: {incident_count}")
        print(f"   - Action items: {action_counts}")
        print(f"   - Explanations: {explanation_counts}")
        return True
        
    except Exception as e:
        print(f"❌ Count function tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_functions_have_docstrings():
    """Verify all functions have proper docstrings."""
    try:
        from api.db_layer import worker_reporting_db
        
        functions_to_check = [
            'get_worker_identity',
            'count_worker_incidents',
            'count_worker_action_items',
            'count_worker_explanation_status'
        ]
        
        for func_name in functions_to_check:
            func = getattr(worker_reporting_db, func_name)
            docstring = inspect.getdoc(func)
            
            assert docstring is not None, f"❌ {func_name} has no docstring"
            assert len(docstring) > 50, f"❌ {func_name} docstring is too short"
        
        print("✅ All functions have proper docstrings")
        return True
        
    except Exception as e:
        print(f"❌ Docstring verification failed: {e}")
        return False


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST TASK D-B2 — WORKER DB AGGREGATION FUNCTIONS")
    print("=" * 70)
    print()
    
    tests = [
        ("File Exists", test_file_exists),
        ("Functions Exist", test_functions_exist),
        ("get_connection Pattern", test_get_connection_pattern),
        ("Queries Are SELECT Only", test_queries_are_select_only),
        ("No Service Imports", test_no_service_imports),
        ("No Pydantic Imports", test_no_pydantic_imports),
        ("Return Types", test_return_types),
        ("Functions Have Docstrings", test_functions_have_docstrings),
        ("get_worker_identity Function", test_get_worker_identity_function),
        ("Count Functions Return Correct Types", test_count_functions_return_correct_types),
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
        print("🎉 DB AGGREGATION LAYER OK — ALL TESTS PASSED")
        return 0
    else:
        print("⚠️  DB AGGREGATION LAYER HAS ISSUES — REVIEW FAILURES ABOVE")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
