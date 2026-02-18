"""
TEST B-B5 — V2 PROFILE CONTRACT CONSISTENCY
Phase B — B-B5 — Verification of standardized profile response contracts

GOAL:
Verify all V2 profile endpoints share the same top-level response structure.

ENDPOINTS TO TEST:
- GET /api/v2/doctors/{id}/profile
- GET /api/v2/patients/{id}/profile  
- GET /api/v2/workers/{id}/profile

TEST APPROACH:
- Verify all endpoints use standardized V2 response schemas
- Check top-level keys consistency (profile, metrics, items, meta)
- Validate meta block structure
- Ensure no schema breaks after normalization
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


def header(msg):
    print(f"\n{'='*70}")
    print(msg)
    print('='*70)

def test_step(msg):
    print(f"\n🔍 {msg}")

def success(msg):
    print(f"✅ {msg}")

def failure(msg):
    print(f"❌ {msg}")
    return False


# ============================================================
# TEST EXECUTION
# ============================================================

header("TEST B-B5 — V2 PROFILE CONTRACT CONSISTENCY")
tests_passed = 0
tests_total = 0

# ------------------------------------------------------------
# TEST 1: Profile Schemas File Exists
# ------------------------------------------------------------
test_step("Profile Schemas File Exists...")
tests_total += 1
try:
    schemas_path = backend_path / "api_v2" / "schemas" / "profile_schemas.py"
    assert schemas_path.exists(), f"Profile schemas file not found: {schemas_path}"
    success("V2 profile schemas file exists")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))

# ------------------------------------------------------------
# TEST 2: Schemas Import Successfully
# ------------------------------------------------------------
test_step("Schemas Import...")
tests_total += 1
try:
    from api_v2.schemas.profile_schemas import (
        EntityMeta,
        DoctorProfileV2Response,
        PatientProfileV2Response,
        WorkerProfileV2Response
    )
    success("V2 profile schemas import successfully")
    tests_passed += 1
except Exception as e:
    failure(f"Failed to import schemas: {e}")

# ------------------------------------------------------------
# TEST 3: EntityMeta Has Required Fields
# ------------------------------------------------------------
test_step("EntityMeta Structure...")
tests_total += 1
try:
    from api_v2.schemas.profile_schemas import EntityMeta
    
    fields = EntityMeta.model_fields if hasattr(EntityMeta, 'model_fields') else EntityMeta.__fields__
    
    required_fields = ['entity_type', 'entity_id', 'period_from', 'period_to']
    for field in required_fields:
        assert field in fields, f"EntityMeta missing field: {field}"
    
    success("EntityMeta has all required fields")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check EntityMeta: {e}")

# ------------------------------------------------------------
# TEST 4: All Profile Schemas Have Same Top-Level Keys
# ------------------------------------------------------------
test_step("Profile Schemas Consistency...")
tests_total += 1
try:
    from api_v2.schemas.profile_schemas import (
        DoctorProfileV2Response,
        PatientProfileV2Response,
        WorkerProfileV2Response
    )
    
    required_top_level_keys = {'profile', 'metrics', 'items', 'meta'}
    
    schemas = [
        ("DoctorProfileV2Response", DoctorProfileV2Response),
        ("PatientProfileV2Response", PatientProfileV2Response),
        ("WorkerProfileV2Response", WorkerProfileV2Response)
    ]
    
    for schema_name, schema_class in schemas:
        fields = schema_class.model_fields if hasattr(schema_class, 'model_fields') else schema_class.__fields__
        schema_keys = set(fields.keys())
        
        assert schema_keys == required_top_level_keys, \
               f"{schema_name} has wrong keys: {schema_keys}, expected {required_top_level_keys}"
    
    success("All profile schemas have consistent top-level keys (profile, metrics, items, meta)")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check schema consistency: {e}")

# ------------------------------------------------------------
# TEST 5: Doctor Router Uses V2 Schema
# ------------------------------------------------------------
test_step("Doctor Router Uses V2 Schema...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import doctors_router as doctor_module
    
    # Get the router source code
    source = inspect.getsource(doctor_module)
    
    # Check for V2 schema import
    assert "DoctorProfileV2Response" in source, "Doctor router doesn't import DoctorProfileV2Response"
    
    # Check for meta block creation
    assert "EntityMeta" in source, "Doctor router doesn't use EntityMeta"
    assert "entity_type=\"doctor\"" in source or 'entity_type="doctor"' in source, \
           "Doctor router doesn't set entity_type to 'doctor'"
    
    success("Doctor router uses V2 response schema")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check doctor router: {e}")

# ------------------------------------------------------------
# TEST 6: Patient Router Uses V2 Schema
# ------------------------------------------------------------
test_step("Patient Router Uses V2 Schema...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import patients_router as patient_module
    
    # Get the router source code
    source = inspect.getsource(patient_module)
    
    # Check for V2 schema import
    assert "PatientProfileV2Response" in source, "Patient router doesn't import PatientProfileV2Response"
    
    # Check for meta block creation
    assert "EntityMeta" in source, "Patient router doesn't use EntityMeta"
    assert "entity_type=\"patient\"" in source or 'entity_type="patient"' in source, \
           "Patient router doesn't set entity_type to 'patient'"
    
    success("Patient router uses V2 response schema")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check patient router: {e}")

# ------------------------------------------------------------
# TEST 7: Worker Router Uses V2 Schema
# ------------------------------------------------------------
test_step("Worker Router Uses V2 Schema...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import workers_router as worker_module
    
    # Get the router source code
    source = inspect.getsource(worker_module)
    
    # Check for V2 schema import
    assert "WorkerProfileV2Response" in source, "Worker router doesn't import WorkerProfileV2Response"
    
    # Check for meta block creation
    assert "EntityMeta" in source, "Worker router doesn't use EntityMeta"
    assert "entity_type=\"worker\"" in source or 'entity_type="worker"' in source, \
           "Worker router doesn't set entity_type to 'worker'"
    
    success("Worker router uses V2 response schema")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check worker router: {e}")

# ------------------------------------------------------------
# TEST 8: Doctor Profile Endpoint Has response_model
# ------------------------------------------------------------
test_step("Doctor Profile Endpoint Has response_model...")
tests_total += 1
try:
    from api_v2.routers.doctors_router import router
    
    # Find the profile endpoint
    profile_endpoint = None
    for route in router.routes:
        if hasattr(route, 'path') and '/profile' in route.path:
            if hasattr(route, 'methods') and 'GET' in route.methods:
                profile_endpoint = route
                break
    
    assert profile_endpoint is not None, "Doctor profile endpoint not found"
    
    # Check response_model
    assert hasattr(profile_endpoint, 'response_model'), "Doctor profile endpoint missing response_model"
    
    response_model_name = profile_endpoint.response_model.__name__ if hasattr(profile_endpoint.response_model, '__name__') else str(profile_endpoint.response_model)
    assert 'DoctorProfileV2Response' in response_model_name, \
           f"Doctor profile endpoint has wrong response_model: {response_model_name}"
    
    success("Doctor profile endpoint has correct response_model")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check doctor endpoint: {e}")

# ------------------------------------------------------------
# TEST 9: Patient Profile Endpoint Has response_model
# ------------------------------------------------------------
test_step("Patient Profile Endpoint Has response_model...")
tests_total += 1
try:
    from api_v2.routers.patients_router import router
    
    # Find the profile endpoint
    profile_endpoint = None
    for route in router.routes:
        if hasattr(route, 'path') and '/profile' in route.path:
            if hasattr(route, 'methods') and 'GET' in route.methods:
                profile_endpoint = route
                break
    
    assert profile_endpoint is not None, "Patient profile endpoint not found"
    
    # Check response_model
    assert hasattr(profile_endpoint, 'response_model'), "Patient profile endpoint missing response_model"
    
    response_model_name = profile_endpoint.response_model.__name__ if hasattr(profile_endpoint.response_model, '__name__') else str(profile_endpoint.response_model)
    assert 'PatientProfileV2Response' in response_model_name, \
           f"Patient profile endpoint has wrong response_model: {response_model_name}"
    
    success("Patient profile endpoint has correct response_model")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check patient endpoint: {e}")

# ------------------------------------------------------------
# TEST 10: Worker Profile Endpoint Exists
# ------------------------------------------------------------
test_step("Worker Profile Endpoint Exists...")
tests_total += 1
try:
    from api_v2.routers.workers_router import router
    
    # Find the profile endpoint
    profile_endpoint = None
    for route in router.routes:
        if hasattr(route, 'path') and '/profile' in route.path:
            if hasattr(route, 'methods') and 'GET' in route.methods:
                profile_endpoint = route
                break
    
    assert profile_endpoint is not None, "Worker profile endpoint not found"
    
    success("Worker profile endpoint exists in V2 router")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check worker endpoint exists: {e}")

# ------------------------------------------------------------
# TEST 11: Worker Profile Endpoint Has response_model
# ------------------------------------------------------------
test_step("Worker Profile Endpoint Has response_model...")
tests_total += 1
try:
    from api_v2.routers.workers_router import router
    
    # Find the profile endpoint
    profile_endpoint = None
    for route in router.routes:
        if hasattr(route, 'path') and '/profile' in route.path:
            if hasattr(route, 'methods') and 'GET' in route.methods:
                profile_endpoint = route
                break
    
    assert profile_endpoint is not None, "Worker profile endpoint not found"
    
    # Check response_model
    assert hasattr(profile_endpoint, 'response_model'), "Worker profile endpoint missing response_model"
    
    response_model_name = profile_endpoint.response_model.__name__ if hasattr(profile_endpoint.response_model, '__name__') else str(profile_endpoint.response_model)
    assert 'WorkerProfileV2Response' in response_model_name, \
           f"Worker profile endpoint has wrong response_model: {response_model_name}"
    
    success("Worker profile endpoint has correct response_model")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check worker endpoint: {e}")

# ------------------------------------------------------------
# TEST 12: All Routers Have Profile Comment
# ------------------------------------------------------------
test_step("Routers Have Normalization Comment...")
tests_total += 1
try:
    import inspect
    from api_v2.routers import doctors_router, patients_router, workers_router
    
    routers = [
        ("doctors_router", doctors_router),
        ("patients_router", patients_router),
        ("workers_router", workers_router)
    ]
    
    comment_found = 0
    for router_name, router_module in routers:
        source = inspect.getsource(router_module)
        if "Phase B — V2 profile contract normalized" in source or \
           "V2 profile contract normalized" in source:
            comment_found += 1
    
    assert comment_found == 3, f"Only {comment_found}/3 routers have normalization comment"
    
    success("All routers have 'Phase B — V2 profile contract normalized' comment")
    tests_passed += 1
except AssertionError as e:
    failure(str(e))
except Exception as e:
    failure(f"Failed to check comments: {e}")

# ============================================================
# SUMMARY
# ============================================================
header("SUMMARY")
print(f"\nTests Passed: {tests_passed}/{tests_total}")

if tests_passed == tests_total:
    print("\n✅ ALL TESTS PASSED — B-B5 CONTRACT CONSISTENCY VERIFIED")
    sys.exit(0)
else:
    print(f"\n❌ {tests_total - tests_passed} TEST(S) FAILED")
    sys.exit(1)
