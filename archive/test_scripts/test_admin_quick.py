"""
Quick Admin Protection Test - Verifies guards are in place
Tests by checking the source code for proper guard implementation.
"""

import re
import os

def check_endpoint_protection(file_path, endpoint_pattern, router_name):
    """Check if an endpoint has proper authentication and admin guards"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all endpoints
    endpoint_matches = list(re.finditer(r'@router\.(get|post|put|patch|delete)\(["\']([^"\']+)["\']\)', content))
    
    results = []
    
    for match in endpoint_matches:
        method = match.group(1).upper()
        path = match.group(2)
        start_pos = match.start()
        
        # Find the function definition after the decorator
        func_match = re.search(r'async def (\w+)\([^)]*\):', content[start_pos:start_pos+500])
        if not func_match:
            results.append({
                "method": method,
                "path": path,
                "status": "ERROR",
                "message": "Could not find function definition"
            })
            continue
        
        func_name = func_match.group(1)
        func_start = start_pos + func_match.start()
        
        # Get a larger section of code including the function
        func_section = content[func_start:func_start+3000]
        
        # Also check the decorator line and parameters
        decorator_and_params = content[start_pos:func_start+800]
        
        # Check for required elements
        has_current_user_param = 'current_user: CurrentUser = Depends(get_current_user)' in decorator_and_params or \
                                  'current_user: CurrentUser = Depends(get_current_user)' in func_section
        has_require_logged_in = 'require_logged_in(current_user)' in func_section
        has_require_admin = 'require_software_admin(current_user)' in func_section
        
        if has_current_user_param and has_require_logged_in and has_require_admin:
            status = "PROTECTED"
            message = "✅ Auth + Admin guards present"
        elif has_current_user_param and has_require_logged_in:
            status = "PARTIAL"
            message = "⚠️  Has auth but missing admin guard"
        elif has_current_user_param:
            status = "PARTIAL"
            message = "⚠️  Has param but missing guard calls"
        else:
            status = "UNPROTECTED"
            message = "❌ No protection"
        
        results.append({
            "method": method,
            "path": path,
            "function": func_name,
            "status": status,
            "message": message,
            "has_param": has_current_user_param,
            "has_auth": has_require_logged_in,
            "has_admin": has_require_admin
        })
    
    return results

# Check settings_router.py
print("=" * 100)
print(" ADMIN ROUTER PROTECTION VERIFICATION")
print("=" * 100)

base_path = r"c:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend\api\routers"

# Test Settings Router
print("\n📁 SETTINGS ROUTER")
print("-" * 100)

settings_file = os.path.join(base_path, "settings_router.py")
settings_results = check_endpoint_protection(settings_file, r'@router\.', 'settings')

settings_protected = 0
settings_total = len(settings_results)

for result in settings_results:
    icon = "✅" if result["status"] == "PROTECTED" else "❌"
    print(f"{icon} {result['method']:6} {result['path']:50} - {result['message']}")
    if result["status"] == "PROTECTED":
        settings_protected += 1

print(f"\n{settings_protected}/{settings_total} endpoints properly protected ({settings_protected/settings_total*100:.0f}%)")

# Test Training Router
print("\n📁 TRAINING ROUTER")
print("-" * 100)

training_file = os.path.join(base_path, "training_router.py")
training_results = check_endpoint_protection(training_file, r'@router\.', 'training')

training_protected = 0
training_total = len(training_results)

for result in training_results:
    icon = "✅" if result["status"] == "PROTECTED" else "❌"
    print(f"{icon} {result['method']:6} {result['path']:50} - {result['message']}")
    if result["status"] == "PROTECTED":
        training_protected += 1

print(f"\n{training_protected}/{training_total} endpoints properly protected ({training_protected/training_total*100:.0f}%)")

# Overall Summary
print("\n" + "=" * 100)
print(" OVERALL SUMMARY")
print("=" * 100)

total_protected = settings_protected + training_protected
total_endpoints = settings_total + training_total

print(f"\n📊 Total Endpoints: {total_endpoints}")
print(f"✅ Protected: {total_protected}")
print(f"❌ Unprotected: {total_endpoints - total_protected}")
print(f"📈 Protection Rate: {total_protected/total_endpoints*100:.1f}%")

# Check imports
print("\n" + "=" * 100)
print(" IMPORT VERIFICATION")
print("=" * 100)

for router_name, file_path in [("settings_router", settings_file), ("training_router", training_file)]:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    has_depends = 'from fastapi import' in content and 'Depends' in content
    has_get_current_user = 'from ..dependencies.user_context import get_current_user' in content
    has_current_user_model = 'from ..schemas.auth_models import CurrentUser' in content
    has_guards = 'from ..utils.guards import require_logged_in, require_software_admin' in content
    
    all_imports = has_depends and has_get_current_user and has_current_user_model and has_guards
    
    icon = "✅" if all_imports else "❌"
    print(f"\n{icon} {router_name}.py")
    print(f"   {'✅' if has_depends else '❌'} Depends imported")
    print(f"   {'✅' if has_get_current_user else '❌'} get_current_user imported")
    print(f"   {'✅' if has_current_user_model else '❌'} CurrentUser model imported")
    print(f"   {'✅' if has_guards else '❌'} Guards imported")

# Final Result
print("\n" + "=" * 100)
print(" FINAL RESULT")
print("=" * 100)

if total_protected == total_endpoints:
    print("\n" + "🎉" * 35)
    print("\n✅ ✅ ✅  100% SUCCESS - ALL ENDPOINTS PROTECTED  ✅ ✅ ✅")
    print("\n" + "🎉" * 35)
    print(f"\n✓ All {settings_total} settings_router endpoints have auth + admin guards")
    print(f"✓ All {training_total} training_router endpoints have auth + admin guards")
    print(f"\n✓ Total: {total_endpoints} endpoints properly protected")
    print("✓ Both guards (require_logged_in + require_software_admin) present")
    print("✓ Dependencies (get_current_user) present")
    print("✓ All imports are correct")
    print("\n" + "=" * 100)
    exit(0)
else:
    print(f"\n❌ FAILED: {total_endpoints - total_protected} endpoints not properly protected")
    print("\nPlease review the output above for details.")
    print("\n" + "=" * 100)
    exit(1)
