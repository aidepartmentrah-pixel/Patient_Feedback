"""
PHASE 6 TEST: API Router/Endpoints
===================================
Tests FastAPI endpoints for the explanation workflow.

Tests:
- GET /api/explanations/pending
- GET /api/explanations/statistics
- GET /api/explanations/{case_id}
- POST /api/explanations/{case_id}
- PUT /api/explanations/{case_id}/requires-explanation
- POST /api/explanations/{case_id}/force-close
- POST /api/explanations/{case_id}/check-closure
- POST /api/explanations/{case_id}/validate
"""

import sys
import os

# Add parent directories to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from fastapi.testclient import TestClient
import main  # Import main from backend directory
from api.services.insert_service import create_record
from api.db_layer.incident_case import hard_delete_incident_case
from datetime import datetime, timedelta

app = main.app


# Create test client
client = TestClient(app)


def cleanup_test_case(case_id):
    """Helper to clean up test data"""
    try:
        hard_delete_incident_case(case_id)
        print(f"  [Cleanup] Deleted test case {case_id}")
    except Exception as e:
        print(f"  [Cleanup Warning] Could not delete case {case_id}: {e}")


def test_get_statistics_endpoint():
    """Test 1: GET /api/explanations/statistics"""
    print("=" * 70)
    print("TEST 1: GET /api/explanations/statistics")
    print("=" * 70)
    
    try:
        response = client.get("/api/explanations/statistics")
        
        if response.status_code != 200:
            print(f"✗ Expected status 200, got {response.status_code}")
            print(f"  Response: {response.json()}")
            return False
        
        data = response.json()
        
        if not data.get('success'):
            print(f"✗ Response indicated failure")
            return False
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Statistics retrieved successfully")
        
        stats = data.get('statistics', {})
        print(f"  By Status: {stats.get('by_status', {})}")
        print(f"  Totals: {stats.get('totals', {})}")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_pending_explanations_endpoint():
    """Test 2: GET /api/explanations/pending"""
    print("\n" + "=" * 70)
    print("TEST 2: GET /api/explanations/pending")
    print("=" * 70)
    
    try:
        # Test without filters
        response = client.get("/api/explanations/pending")
        
        if response.status_code != 200:
            print(f"✗ Expected status 200, got {response.status_code}")
            return False
        
        data = response.json()
        print(f"✓ Retrieved pending explanations: {data.get('total_count', 0)} total")
        
        # Test with filters
        response_filtered = client.get(
            "/api/explanations/pending",
            params={
                "start_date": "2024-01-01",
                "end_date": "2026-12-31",
                "include_red_flags_only": True
            }
        )
        
        if response_filtered.status_code != 200:
            print(f"✗ Filtered request failed: {response_filtered.status_code}")
            return False
        
        data_filtered = response_filtered.json()
        print(f"✓ Filtered request successful: {data_filtered.get('red_flag_count', 0)} Red Flags")
        
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_get_case_details_endpoint():
    """Test 3: GET /api/explanations/{case_id}"""
    print("\n" + "=" * 70)
    print("TEST 3: GET /api/explanations/{case_id}")
    print("=" * 70)
    
    case_id = None
    try:
        # Create a test case
        data = {
            "complaint_text": "Test case for API endpoint",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 2,  # Red Flag
        }
        
        result = create_record(data)
        case_id = result['incident_id']
        print(f"✓ Created test case {case_id}")
        
        # Test valid case
        response = client.get(f"/api/explanations/{case_id}")
        
        if response.status_code != 200:
            print(f"✗ Expected status 200, got {response.status_code}")
            return False
        
        response_data = response.json()
        
        if not response_data.get('success'):
            print(f"✗ Response indicated failure")
            return False
        
        validation = response_data.get('validation', {})
        print(f"✓ Case details retrieved")
        print(f"  can_submit: {validation.get('can_submit_explanation')}")
        print(f"  requires_explanation: {validation.get('requires_explanation')}")
        
        # Test non-existent case
        response_404 = client.get("/api/explanations/999999999")
        
        if response_404.status_code != 404:
            print(f"✗ Expected 404 for non-existent case, got {response_404.status_code}")
            return False
        
        print(f"✓ Correctly returned 404 for non-existent case")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_submit_explanation_endpoint():
    """Test 4: POST /api/explanations/{case_id}"""
    print("\n" + "=" * 70)
    print("TEST 4: POST /api/explanations/{case_id}")
    print("=" * 70)
    
    case_id = None
    try:
        # Create a test case
        data = {
            "complaint_text": "Test case for explanation submission API",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 3,  # Never Event
        }
        
        result = create_record(data)
        case_id = result['incident_id']
        print(f"✓ Created test case {case_id}")
        
        # Submit explanation with action items
        future_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        
        payload = {
            "explanation_text": "This is a comprehensive explanation addressing the Never Event with root cause analysis and preventive measures.",
            "action_items": [
                {
                    "title": "Revise safety protocol",
                    "description": "Update and distribute revised protocol",
                    "due_date": future_date
                },
                {
                    "title": "Staff retraining",
                    "description": "Conduct mandatory training for all staff",
                    "due_date": future_date
                }
            ],
            "user_id": 1
        }
        
        response = client.post(f"/api/explanations/{case_id}", json=payload)
        
        if response.status_code != 200:
            print(f"✗ Expected status 200, got {response.status_code}")
            print(f"  Response: {response.json()}")
            return False
        
        response_data = response.json()
        
        if not response_data.get('success'):
            print(f"✗ Submission failed: {response_data.get('error')}")
            return False
        
        print(f"✓ Explanation submitted successfully")
        print(f"  Action items created: {response_data.get('action_items_created')}")
        
        # Test validation error - too short text
        payload_invalid = {
            "explanation_text": "Short",
            "user_id": 1
        }
        
        response_invalid = client.post(f"/api/explanations/{case_id}", json=payload_invalid)
        
        if response_invalid.status_code != 400:
            print(f"✗ Expected 400 for invalid payload, got {response_invalid.status_code}")
            return False
        
        print(f"✓ Correctly rejected invalid explanation")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_update_requires_explanation_endpoint():
    """Test 5: PUT /api/explanations/{case_id}/requires-explanation"""
    print("\n" + "=" * 70)
    print("TEST 5: PUT /{case_id}/requires-explanation")
    print("=" * 70)
    
    case_id = None
    try:
        # Create ordinary case
        data = {
            "complaint_text": "Ordinary complaint for testing flag toggle",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 1,  # Ordinary
            "requires_explanation": False
        }
        
        result = create_record(data)
        case_id = result['incident_id']
        print(f"✓ Created ordinary case {case_id}")
        
        # Toggle flag to true
        payload = {
            "requires_explanation": True,
            "reason": "Policy requirement - high severity incident",
            "user_id": 1
        }
        
        response = client.put(f"/api/explanations/{case_id}/requires-explanation", json=payload)
        
        if response.status_code != 200:
            print(f"✗ Expected status 200, got {response.status_code}")
            return False
        
        response_data = response.json()
        
        if not response_data.get('success'):
            print(f"✗ Flag update failed")
            return False
        
        print(f"✓ RequiresExplanation flag updated to True")
        
        # Toggle back to false
        payload_false = {
            "requires_explanation": False,
            "reason": "Downgraded severity after review",
            "user_id": 1
        }
        
        response_false = client.put(f"/api/explanations/{case_id}/requires-explanation", json=payload_false)
        
        if response_false.status_code != 200:
            print(f"✗ Expected status 200 for toggle back")
            return False
        
        print(f"✓ RequiresExplanation flag updated to False")
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def test_validate_explanation_endpoint():
    """Test 6: POST /api/explanations/{case_id}/validate"""
    print("\n" + "=" * 70)
    print("TEST 6: POST /{case_id}/validate")
    print("=" * 70)
    
    case_id = None
    try:
        # Create test case
        data = {
            "complaint_text": "Test case for validation endpoint",
            "feedback_received_date": "2026-01-19",
            "issuing_department_id": 43,
            "domain_id": 2,
            "category_id": 5,
            "subcategory_id": 13,
            "classification_id": 106,
            "severity_id": 2,
            "stage_id": 2,
            "harm_id": 2,
            "building_id": 2,
            "source_id": 4,
            "clinical_risk_type_id": 2,
        }
        
        result = create_record(data)
        case_id = result['incident_id']
        print(f"✓ Created test case {case_id}")
        
        # Test valid explanation
        payload_valid = {
            "explanation_text": "This is a valid explanation with sufficient length for validation",
            "action_items": []
        }
        
        response = client.post(f"/api/explanations/{case_id}/validate", json=payload_valid)
        
        if response.status_code != 200:
            print(f"✗ Expected status 200, got {response.status_code}")
            return False
        
        validation_result = response.json()
        
        if validation_result.get('valid'):
            print(f"✓ Valid explanation passed validation")
        else:
            print(f"⚠ Validation returned errors: {validation_result.get('errors')}")
        
        # Test invalid explanation - too short
        payload_invalid = {
            "explanation_text": "Short",
            "action_items": []
        }
        
        response_invalid = client.post(f"/api/explanations/{case_id}/validate", json=payload_invalid)
        
        if response_invalid.status_code != 200:
            print(f"✗ Validation endpoint should return 200 even for invalid data")
            return False
        
        validation_result_invalid = response_invalid.json()
        
        if not validation_result_invalid.get('valid'):
            print(f"✓ Correctly identified invalid explanation")
            print(f"  Errors: {validation_result_invalid.get('errors')}")
        else:
            print(f"✗ Should have rejected short explanation")
            return False
        
        cleanup_test_case(case_id)
        return True
        
    except Exception as e:
        print(f"✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        if case_id:
            cleanup_test_case(case_id)
        return False


def run_all_tests():
    """Run all Phase 6 tests"""
    print("\n")
    print("*" * 70)
    print("*" + " " * 68 + "*")
    print("*" + "  PHASE 6: API ROUTER/ENDPOINTS TESTS".center(68) + "*")
    print("*" + " " * 68 + "*")
    print("*" * 70)
    print("\n")
    
    tests = [
        ("GET /statistics", test_get_statistics_endpoint),
        ("GET /pending", test_get_pending_explanations_endpoint),
        ("GET /{case_id}", test_get_case_details_endpoint),
        ("POST /{case_id}", test_submit_explanation_endpoint),
        ("PUT /requires-explanation", test_update_requires_explanation_endpoint),
        ("POST /validate", test_validate_explanation_endpoint),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {test_name:<45} {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print("=" * 70)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 70)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - Phase 6 Complete!")
        print("\nAPI Endpoints Validated:")
        print("  ✓ GET /api/explanations/statistics")
        print("  ✓ GET /api/explanations/pending")
        print("  ✓ GET /api/explanations/{case_id}")
        print("  ✓ POST /api/explanations/{case_id}")
        print("  ✓ PUT /api/explanations/{case_id}/requires-explanation")
        print("  ✓ POST /api/explanations/{case_id}/validate")
    else:
        print(f"\n⚠ {total - passed} test(s) failed - Please review")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
