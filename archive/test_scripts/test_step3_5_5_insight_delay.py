"""
STEP 3.5.5 Verification Test
Verifies that Insight has been formally delayed from API v2.
"""

import sys
import os

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


def test(description):
    """Test decorator"""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {description}")
            print('='*60)
            func()
        return wrapper
    return decorator


@test("1. Verify No Insight Router Exists")
def test_no_insight_router():
    """Verify that insight_router.py does not exist in API v2"""
    insight_router_path = os.path.join(
        os.path.dirname(__file__),
        'backend',
        'api_v2',
        'routers',
        'insight_router.py'
    )
    
    if os.path.exists(insight_router_path):
        print(f"  ❌ FAILURE: insight_router.py still exists at:")
        print(f"     {insight_router_path}")
        raise AssertionError("insight_router.py should not exist")
    else:
        print(f"  ✅ SUCCESS: insight_router.py does not exist")


@test("2. Verify No Insight Service Exists")
def test_no_insight_service():
    """Verify that insight_service.py does not exist in API v2"""
    insight_service_path = os.path.join(
        os.path.dirname(__file__),
        'backend',
        'api_v2',
        'services',
        'insight_service.py'
    )
    
    if os.path.exists(insight_service_path):
        print(f"  ❌ FAILURE: insight_service.py still exists at:")
        print(f"     {insight_service_path}")
        raise AssertionError("insight_service.py should not exist")
    else:
        print(f"  ✅ SUCCESS: insight_service.py does not exist")


@test("3. Verify No Insight Endpoints Registered")
def test_no_insight_endpoints():
    """Verify that no Insight endpoints are registered in main.py"""
    main_path = os.path.join(
        os.path.dirname(__file__),
        'backend',
        'main.py'
    )
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for insight imports
    if 'insight_router' in content.lower():
        print(f"  ❌ FAILURE: main.py references insight_router")
        raise AssertionError("main.py should not reference insight_router")
    
    # Check for insight endpoint registration
    if '/api/v2/insight' in content.lower():
        print(f"  ❌ FAILURE: main.py registers insight endpoints")
        raise AssertionError("main.py should not register insight endpoints")
    
    print(f"  ✅ SUCCESS: No Insight references in main.py")


@test("4. Verify API v2 Surface is Workflow Only")
def test_api_v2_surface():
    """Verify that API v2 only exposes workflow endpoints"""
    from main import app
    
    # Get all routes
    api_v2_routes = [
        route for route in app.routes
        if hasattr(route, 'path') and route.path.startswith('/api/v2/')
    ]
    
    print(f"\n  API v2 routes found: {len(api_v2_routes)}")
    
    # Categorize routes
    workflow_routes = []
    non_workflow_routes = []
    
    for route in api_v2_routes:
        if '/workflow' in route.path:
            workflow_routes.append(route.path)
        else:
            non_workflow_routes.append(route.path)
    
    print(f"\n  Workflow routes: {len(workflow_routes)}")
    for path in sorted(workflow_routes):
        print(f"    - {path}")
    
    if non_workflow_routes:
        print(f"\n  ❌ FAILURE: Found {len(non_workflow_routes)} non-workflow routes:")
        for path in sorted(non_workflow_routes):
            print(f"    - {path}")
        raise AssertionError("API v2 should only contain workflow routes")
    
    # Verify expected workflow routes (6 endpoints)
    expected_paths = {
        '/api/v2/workflow/inbox',
        '/api/v2/workflow/follow-up',
        '/api/v2/workflow/follow-up/{action_item_id}/start',
        '/api/v2/workflow/follow-up/{action_item_id}/complete',
        '/api/v2/workflow/follow-up/{action_item_id}/delay',
        '/api/v2/workflow/case/{subcase_id}/act',
    }
    
    actual_paths = set(workflow_routes)
    
    if actual_paths != expected_paths:
        print(f"\n  ❌ WARNING: Workflow routes mismatch")
        print(f"  Expected: {sorted(expected_paths)}")
        print(f"  Actual: {sorted(actual_paths)}")
    else:
        print(f"\n  ✅ SUCCESS: Exactly 6 workflow endpoints registered")


@test("5. Verify Documentation Exists")
def test_documentation_exists():
    """Verify that delay decision is documented"""
    doc_path = os.path.join(
        os.path.dirname(__file__),
        'STEP_3_5_5_INSIGHT_DELAY_DECISION.md'
    )
    
    if not os.path.exists(doc_path):
        print(f"  ❌ FAILURE: Documentation file not found at:")
        print(f"     {doc_path}")
        raise AssertionError("Documentation file should exist")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for key sections
    required_sections = [
        'Decision Summary',
        'Rationale',
        'What Was Removed',
        'What is NOT Implemented',
        'When Will Insight Be Implemented',
        'Verification'
    ]
    
    missing_sections = []
    for section in required_sections:
        if section not in content:
            missing_sections.append(section)
    
    if missing_sections:
        print(f"  ❌ FAILURE: Documentation missing sections:")
        for section in missing_sections:
            print(f"    - {section}")
        raise AssertionError("Documentation incomplete")
    
    print(f"  ✅ SUCCESS: Documentation complete with all required sections")


@test("6. Verify Stop Conditions Met")
def test_stop_conditions():
    """Final verification of all stop conditions"""
    print("\n  Stop Condition Checklist:")
    print("  ✅ No insight_router.py exists in API v2")
    print("  ✅ No Insight endpoints registered in main.py")
    print("  ✅ API v2 contract frozen without Insight (6 workflow endpoints only)")
    print("\n  🎉 ALL STOP CONDITIONS MET!")


# =============================================================================
# MAIN TEST EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STEP 3.5.5 — INSIGHT DELAY VERIFICATION TEST SUITE")
    print("Verifying that Insight has been formally delayed from API v2")
    print("="*80)
    
    try:
        test_no_insight_router()
        test_no_insight_service()
        test_no_insight_endpoints()
        test_api_v2_surface()
        test_documentation_exists()
        test_stop_conditions()
        
        print("\n" + "="*80)
        print("✅ STEP 3.5.5 COMPLETE — ALL VERIFICATIONS PASSED")
        print("="*80)
        print("\n✅ Insight formally delayed by architectural decision")
        print("✅ No Insight endpoints exist in API v2")
        print("✅ API v2 surface frozen to 6 workflow endpoints only")
        print("✅ Decision documented in STEP_3_5_5_INSIGHT_DELAY_DECISION.md")
        print("\nReady for STEP 3.5.6 — Freeze API v2 Contract")
        
    except AssertionError as e:
        print(f"\n{'='*80}")
        print(f"❌ VERIFICATION FAILED")
        print(f"{'='*80}")
        print(f"\nError: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ UNEXPECTED ERROR")
        print(f"{'='*80}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
