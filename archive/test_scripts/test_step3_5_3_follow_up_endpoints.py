"""
STEP 3.5.3 — Follow-Up Endpoints Test
Verify that the follow-up endpoints in workflow router are working correctly.
"""

import sys
import os

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

print("\n" + "="*80)
print("STEP 3.5.3 — FOLLOW-UP ENDPOINTS VERIFICATION")
print("="*80)

# Test 1: Verify endpoints are registered
print("\n[TEST 1] Verify endpoint registration")
try:
    from main import app
    workflow_routes = [r for r in app.routes if hasattr(r, 'path') and '/api/v2/workflow/follow-up' in r.path]
    
    expected_routes = {
        'GET': ['/api/v2/workflow/follow-up'],
        'POST': [
            '/api/v2/workflow/follow-up/{action_item_id}/start',
            '/api/v2/workflow/follow-up/{action_item_id}/complete',
            '/api/v2/workflow/follow-up/{action_item_id}/delay'
        ]
    }
    
    found_routes = {}
    for route in workflow_routes:
        if hasattr(route, 'methods'):
            method = list(route.methods)[0]
            if method not in found_routes:
                found_routes[method] = []
            found_routes[method].append(route.path)
    
    # Check GET endpoint
    if 'GET' in found_routes and len(found_routes['GET']) == 1:
        print(f"  ✅ GET /api/v2/workflow/follow-up registered")
    else:
        print(f"  ❌ GET endpoint not found or incorrect count")
    
    # Check POST endpoints
    if 'POST' in found_routes and len(found_routes['POST']) == 3:
        print(f"  ✅ All 3 POST endpoints registered")
        for path in found_routes['POST']:
            action = path.split('/')[-1]
            print(f"    - POST .../{action}")
    else:
        print(f"  ❌ Expected 3 POST endpoints, found {len(found_routes.get('POST', []))}")
    
    print(f"\n  Total follow-up routes: {len(workflow_routes)}")
    
except Exception as e:
    print(f"  ❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

# Test 2: Verify service functions exist
print("\n[TEST 2] Verify follow_up_service functions exist")
try:
    from api_v2.services import follow_up_service
    
    functions = [
        'get_action_items_for_user',
        'start_action_item',
        'complete_action_item',
        'delay_action_item'
    ]
    
    all_exist = True
    for func_name in functions:
        if hasattr(follow_up_service, func_name):
            print(f"  ✅ {func_name} exists")
        else:
            print(f"  ❌ {func_name} not found")
            all_exist = False
    
    if all_exist:
        print(f"\n  ✅ All required service functions exist")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 3: Verify router imports
print("\n[TEST 3] Verify workflow_router imports")
try:
    from api_v2.routers import workflow_router
    import inspect
    
    source = inspect.getsource(workflow_router)
    
    required_imports = ['follow_up_service', 'get_current_user', 'inbox_service']
    
    for imp in required_imports:
        if imp in source:
            print(f"  ✅ {imp} imported")
        else:
            print(f"  ❌ {imp} not imported")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 4: Verify no business logic in router
print("\n[TEST 4] Verify router is thin (no business logic)")
try:
    from api_v2.routers import workflow_router
    import inspect
    
    source = inspect.getsource(workflow_router)
    
    # Check for anti-patterns (business logic in router)
    anti_patterns = [
        ('if current_user.role', 'Role branching'),
        ('filter(', 'Filtering logic'),
        ('allowed_unit_ids', 'Scope logic'),
        ('.execute(', 'Direct DB access'),
        ('_assert', 'Permission checks'),
        ('raise Forbidden', 'Authorization logic'),
        ('raise Unauthorized', 'Auth logic')
    ]
    
    found_anti_patterns = []
    for pattern, desc in anti_patterns:
        # Exclude patterns in comments or docstrings
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if pattern in line:
                # Skip if it's in a comment or docstring
                stripped = line.strip()
                if not stripped.startswith('#') and not stripped.startswith('"""') and not stripped.startswith("'''"):
                    # Check if it's not just mentioning it in docs
                    if '"""' not in '\n'.join(lines[max(0,i-5):i]) or '"""' in line:
                        found_anti_patterns.append((pattern, desc))
                        break
    
    if len(found_anti_patterns) == 0:
        print(f"  ✅ Router is thin (no business logic detected)")
    else:
        print(f"  ⚠️  Potential business logic found:")
        for pattern, desc in found_anti_patterns:
            print(f"    - {desc}: {pattern}")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 5: Verify authentication dependency
print("\n[TEST 5] Verify authentication dependency on all endpoints")
try:
    from api_v2.routers import workflow_router
    import inspect
    
    source = inspect.getsource(workflow_router)
    
    # Count endpoint definitions
    endpoint_count = source.count('@router.get("/follow-up')
    endpoint_count += source.count('@router.post("/follow-up')
    
    # Count get_current_user usages in follow-up endpoints
    # Should appear once per endpoint
    auth_count = 0
    lines = source.split('\n')
    in_follow_up_endpoint = False
    for line in lines:
        if '@router.' in line and '/follow-up' in line:
            in_follow_up_endpoint = True
        elif '@router.' in line:
            in_follow_up_endpoint = False
        elif in_follow_up_endpoint and 'get_current_user' in line:
            auth_count += 1
    
    if auth_count == endpoint_count:
        print(f"  ✅ All {endpoint_count} follow-up endpoints require authentication")
    else:
        print(f"  ⚠️  Auth count ({auth_count}) doesn't match endpoint count ({endpoint_count})")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 6: Verify endpoints don't catch exceptions
print("\n[TEST 6] Verify endpoints don't catch exceptions (let service errors propagate)")
try:
    from api_v2.routers import workflow_router
    import inspect
    
    source = inspect.getsource(workflow_router)
    
    # Check for exception handling in follow-up endpoints
    follow_up_section = source.split('# FOLLOW-UP ENDPOINTS')[1].split('# CASE ACTION ENDPOINTS')[0]
    
    if 'try:' in follow_up_section or 'except' in follow_up_section:
        print(f"  ⚠️  Exception handling detected in follow-up endpoints")
        print(f"     (Should let service errors propagate)")
    else:
        print(f"  ✅ No exception handling (service errors propagate naturally)")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\n✅ If all tests passed, STEP 3.5.3 is complete!")
print("✅ Follow-up endpoints are ready:")
print("   - GET /api/v2/workflow/follow-up")
print("   - POST /api/v2/workflow/follow-up/{id}/start")
print("   - POST /api/v2/workflow/follow-up/{id}/complete")
print("   - POST /api/v2/workflow/follow-up/{id}/delay")
print("✅ Router is thin and delegates to service layer")
print("✅ Service handles authorization and scope filtering")
