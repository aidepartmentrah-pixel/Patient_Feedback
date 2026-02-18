"""
TEST TASK D-B1 — WORKER PROFILE ENDPOINT CONTRACT

Verifies worker reporting schema definitions are correct and complete.
"""

import sys
import os
from pathlib import Path
from datetime import date

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_schema_file_exists():
    """Verify schema file exists at correct location."""
    schema_path = backend_path / "api" / "schemas" / "worker_reporting_schema.py"
    assert schema_path.exists(), f"❌ Schema file not found at: {schema_path}"
    print("✅ Schema file exists")
    return True


def test_models_compile():
    """Verify models can be imported without errors."""
    try:
        from api.schemas.worker_reporting_schema import (
            WorkerIdentityBlock,
            WorkerMetricBlock,
            WorkerProfileResponse
        )
        print("✅ All models import successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_models_are_pydantic():
    """Verify models are pure Pydantic BaseModel subclasses."""
    from pydantic import BaseModel
    from api.schemas.worker_reporting_schema import (
        WorkerIdentityBlock,
        WorkerMetricBlock,
        WorkerProfileResponse
    )
    
    models = [WorkerIdentityBlock, WorkerMetricBlock, WorkerProfileResponse]
    for model in models:
        assert issubclass(model, BaseModel), f"❌ {model.__name__} is not a Pydantic BaseModel"
    
    print("✅ All models are Pydantic BaseModel subclasses")
    return True


def test_no_db_imports():
    """Verify schema file has no database layer imports."""
    schema_path = backend_path / "api" / "schemas" / "worker_reporting_schema.py"
    with open(schema_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    forbidden_imports = [
        'from ..db_layer',
        'from api.db_layer',
        'import db_layer',
        'from ..services',
        'from api.services',
        'import services',
        'import sqlalchemy',
        'from sqlalchemy'
    ]
    
    for forbidden in forbidden_imports:
        if forbidden in content:
            print(f"❌ Forbidden import found: {forbidden}")
            return False
    
    print("✅ No database or service layer imports found")
    return True


def test_worker_identity_block_fields():
    """Verify WorkerIdentityBlock has correct fields."""
    from api.schemas.worker_reporting_schema import WorkerIdentityBlock
    
    required_fields = {
        'employee_id': int,
        'full_name': str,
        'job_title': (type(None), str),  # Optional[str]
        'department_id': (type(None), int),  # Optional[int]
        'section_id': (type(None), int),  # Optional[int]
        'administration_id': (type(None), int),  # Optional[int]
        'is_active': (type(None), bool)  # Optional[bool]
    }
    
    # Get model fields
    model_fields = WorkerIdentityBlock.model_fields
    
    # Check all required fields exist
    for field_name in required_fields.keys():
        assert field_name in model_fields, f"❌ Missing field: {field_name}"
    
    print("✅ WorkerIdentityBlock has all required fields")
    
    # Test instantiation
    worker = WorkerIdentityBlock(
        employee_id=12345,
        full_name="Test Worker",
        job_title="Test Job",
        department_id=1,
        section_id=2,
        administration_id=3,
        is_active=True
    )
    
    assert worker.employee_id == 12345
    assert worker.full_name == "Test Worker"
    print("✅ WorkerIdentityBlock instantiation works correctly")
    return True


def test_worker_metric_block_fields():
    """Verify WorkerMetricBlock has correct fields."""
    from api.schemas.worker_reporting_schema import WorkerMetricBlock
    
    required_fields = {
        'total_incidents': int,
        'total_action_items': int,
        'completed_action_items': int,
        'overdue_action_items': int,
        'explanation_rejected_count': int,
        'explanation_accepted_count': int
    }
    
    # Get model fields
    model_fields = WorkerMetricBlock.model_fields
    
    # Check all required fields exist
    for field_name in required_fields.keys():
        assert field_name in model_fields, f"❌ Missing field: {field_name}"
    
    print("✅ WorkerMetricBlock has all required fields")
    
    # Test instantiation with defaults
    metrics = WorkerMetricBlock()
    assert metrics.total_incidents == 0
    assert metrics.total_action_items == 0
    print("✅ WorkerMetricBlock defaults to zero for all metrics")
    
    # Test instantiation with data
    metrics2 = WorkerMetricBlock(
        total_incidents=10,
        total_action_items=25,
        completed_action_items=20,
        overdue_action_items=5,
        explanation_rejected_count=2,
        explanation_accepted_count=8
    )
    assert metrics2.total_incidents == 10
    assert metrics2.completed_action_items == 20
    print("✅ WorkerMetricBlock instantiation works correctly")
    return True


def test_worker_profile_response_fields():
    """Verify WorkerProfileResponse has correct fields and structure."""
    from api.schemas.worker_reporting_schema import (
        WorkerProfileResponse,
        WorkerIdentityBlock,
        WorkerMetricBlock
    )
    
    required_fields = {
        'worker': WorkerIdentityBlock,
        'metrics': WorkerMetricBlock,
        'period_from': (type(None), date),  # Optional[date]
        'period_to': (type(None), date)  # Optional[date]
    }
    
    # Get model fields
    model_fields = WorkerProfileResponse.model_fields
    
    # Check all required fields exist
    for field_name in required_fields.keys():
        assert field_name in model_fields, f"❌ Missing field: {field_name}"
    
    print("✅ WorkerProfileResponse has all required fields")
    
    # Test instantiation
    response = WorkerProfileResponse(
        worker=WorkerIdentityBlock(
            employee_id=12345,
            full_name="Test Worker",
            job_title=None,
            department_id=1,
            section_id=None,
            administration_id=None,
            is_active=True
        ),
        metrics=WorkerMetricBlock(
            total_incidents=5,
            total_action_items=10,
            completed_action_items=8,
            overdue_action_items=2,
            explanation_rejected_count=1,
            explanation_accepted_count=4
        ),
        period_from=date(2025, 1, 1),
        period_to=date(2025, 12, 31)
    )
    
    assert response.worker.employee_id == 12345
    assert response.metrics.total_incidents == 5
    assert response.period_from == date(2025, 1, 1)
    print("✅ WorkerProfileResponse instantiation works correctly")
    
    # Test with None dates
    response2 = WorkerProfileResponse(
        worker=WorkerIdentityBlock(
            employee_id=99999,
            full_name="Another Worker"
        ),
        metrics=WorkerMetricBlock(),
        period_from=None,
        period_to=None
    )
    assert response2.period_from is None
    assert response2.period_to is None
    print("✅ WorkerProfileResponse works with None dates (all-time metrics)")
    return True


def test_json_serialization():
    """Verify models can be serialized to JSON."""
    from api.schemas.worker_reporting_schema import (
        WorkerProfileResponse,
        WorkerIdentityBlock,
        WorkerMetricBlock
    )
    
    response = WorkerProfileResponse(
        worker=WorkerIdentityBlock(
            employee_id=12345,
            full_name="Test Worker",
            job_title="Quality Specialist",
            department_id=42,
            section_id=8,
            administration_id=3,
            is_active=True
        ),
        metrics=WorkerMetricBlock(
            total_incidents=12,
            total_action_items=45,
            completed_action_items=38,
            overdue_action_items=3,
            explanation_rejected_count=2,
            explanation_accepted_count=15
        ),
        period_from=date(2025, 1, 1),
        period_to=date(2025, 12, 31)
    )
    
    # Serialize to dict
    data = response.model_dump()
    assert data['worker']['employee_id'] == 12345
    assert data['metrics']['total_incidents'] == 12
    print("✅ Models can be serialized to dict")
    
    # Serialize to JSON
    json_str = response.model_dump_json()
    assert '"employee_id":12345' in json_str.replace(' ', '')
    print("✅ Models can be serialized to JSON string")
    return True


def run_all_tests():
    """Run all verification tests."""
    print("=" * 70)
    print("TEST TASK D-B1 — WORKER PROFILE ENDPOINT CONTRACT")
    print("=" * 70)
    print()
    
    tests = [
        ("File Exists", test_schema_file_exists),
        ("Models Compile", test_models_compile),
        ("Models Are Pydantic", test_models_are_pydantic),
        ("No DB Imports", test_no_db_imports),
        ("WorkerIdentityBlock Fields", test_worker_identity_block_fields),
        ("WorkerMetricBlock Fields", test_worker_metric_block_fields),
        ("WorkerProfileResponse Fields", test_worker_profile_response_fields),
        ("JSON Serialization", test_json_serialization),
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
        print("🎉 SCHEMA CONTRACT OK — ALL TESTS PASSED")
        return 0
    else:
        print("⚠️  SCHEMA CONTRACT HAS ISSUES — REVIEW FAILURES ABOVE")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
