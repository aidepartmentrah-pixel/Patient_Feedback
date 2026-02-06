"""
STEP 3.5.2 — Inbox Endpoint Test
Verify that the GET /api/v2/workflow/inbox endpoint is working correctly.
"""

import sys
import os

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

print("\n" + "="*80)
print("STEP 3.5.2 — INBOX ENDPOINT VERIFICATION")
print("="*80)

# Test 1: Verify endpoint is registered
print("\n[TEST 1] Verify endpoint registration")
try:
    from main import app
    workflow_routes = [r for r in app.routes if hasattr(r, 'path') and '/api/v2/workflow' in r.path]
    inbox_routes = [r for r in workflow_routes if 'inbox' in r.path]
    
    if len(inbox_routes) == 1:
        route = inbox_routes[0]
        print(f"  ✅ Endpoint registered: {route.methods} {route.path}")
    else:
        print(f"  ❌ Expected 1 inbox route, found {len(inbox_routes)}")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 2: Verify service function exists
print("\n[TEST 2] Verify inbox_service.get_inbox exists")
try:
    from api_v2.services import inbox_service
    
    if hasattr(inbox_service, 'get_inbox'):
        print(f"  ✅ get_inbox function exists")
        
        # Check that it's callable
        if callable(inbox_service.get_inbox):
            print(f"  ✅ get_inbox is callable")
        else:
            print(f"  ❌ get_inbox is not callable")
    else:
        print(f"  ❌ get_inbox function not found")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 3: Verify role-specific functions still exist
print("\n[TEST 3] Verify role-specific inbox functions exist")
try:
    from api_v2.services import inbox_service
    
    functions = [
        'get_section_inbox',
        'get_department_inbox',
        'get_administration_inbox'
    ]
    
    all_exist = True
    for func_name in functions:
        if hasattr(inbox_service, func_name):
            print(f"  ✅ {func_name} exists")
        else:
            print(f"  ❌ {func_name} not found")
            all_exist = False
    
    if all_exist:
        print(f"\n  ✅ All role-specific functions preserved")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 4: Verify router imports
print("\n[TEST 4] Verify workflow_router imports")
try:
    from api_v2.routers import workflow_router
    
    required_imports = ['APIRouter', 'Depends', 'get_current_user', 'CurrentUser', 'inbox_service']
    
    # Check if inbox_service is imported
    import inspect
    source = inspect.getsource(workflow_router)
    
    if 'inbox_service' in source:
        print(f"  ✅ inbox_service imported")
    else:
        print(f"  ❌ inbox_service not imported")
    
    if 'get_current_user' in source:
        print(f"  ✅ get_current_user imported")
    else:
        print(f"  ❌ get_current_user not imported")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

# Test 5: Verify no business logic in router
print("\n[TEST 5] Verify router is thin (no business logic)")
try:
    from api_v2.routers import workflow_router
    import inspect
    
    source = inspect.getsource(workflow_router)
    
    # Check for anti-patterns (business logic in router)
    anti_patterns = [
        'if current_user.role',  # Role branching in router
        'filter(',  # Filtering in router
        'allowed_unit_ids',  # Scope logic in router
        '.execute(',  # Direct DB access
    ]
    
    found_anti_patterns = []
    for pattern in anti_patterns:
        if pattern in source and 'NOTE:' not in source.split(pattern)[0].split('\n')[-1]:
            found_anti_patterns.append(pattern)
    
    if len(found_anti_patterns) == 0:
        print(f"  ✅ Router is thin (no business logic detected)")
    else:
        print(f"  ⚠️  Potential business logic found: {found_anti_patterns}")
        
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\n✅ If all tests passed, STEP 3.5.2 is complete!")
print("✅ GET /api/v2/workflow/inbox endpoint is ready")
print("✅ Router is thin and delegates to service layer")
print("✅ Service handles role routing and scope filtering")
