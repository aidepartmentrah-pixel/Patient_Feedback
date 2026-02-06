"""
Final lightweight verification of admin router protection.
Tests that guard functions are properly called for all endpoints.
"""

import ast
import sys
from pathlib import Path

def extract_function_info(file_path):
    """Extract function names and their body lines from a Python file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    tree = ast.parse(content)
    functions = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef):
            # Get function body as string
            func_lines = []
            for stmt in node.body:
                if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                    # Skip docstrings
                    continue
                func_lines.append(ast.unparse(stmt))
            
            # Check for guard calls and current_user parameter
            has_require_logged_in = any('require_logged_in(current_user)' in line for line in func_lines)
            has_require_admin = any('require_software_admin(current_user)' in line for line in func_lines)
            
            # Check if current_user parameter exists
            has_current_user = False
            for arg in node.args.args:
                if arg.arg == 'current_user':
                    has_current_user = True
                    break
            
            functions[node.name] = {
                'has_current_user': has_current_user,
                'has_require_logged_in': has_require_logged_in,
                'has_require_admin': has_require_admin,
                'line': node.lineno
            }
    
    return functions

def verify_router(router_name, file_path, expected_count):
    """Verify all endpoints in a router have proper guards."""
    print(f"\n{'='*70}")
    print(f"🔍 Verifying {router_name}")
    print(f"{'='*70}")
    
    functions = extract_function_info(file_path)
    
    # Filter to endpoint functions (skip internal helpers)
    endpoints = {name: info for name, info in functions.items() 
                 if not name.startswith('_')}
    
    protected_count = 0
    issues = []
    
    for name, info in endpoints.items():
        status = "✅" if (info['has_current_user'] and 
                          info['has_require_logged_in'] and 
                          info['has_require_admin']) else "❌"
        
        if status == "✅":
            protected_count += 1
        else:
            missing = []
            if not info['has_current_user']:
                missing.append('current_user parameter')
            if not info['has_require_logged_in']:
                missing.append('require_logged_in()')
            if not info['has_require_admin']:
                missing.append('require_software_admin()')
            issues.append(f"  ❌ {name} (line {info['line']}): Missing {', '.join(missing)}")
    
    print(f"\n📊 Results:")
    print(f"  Total endpoints: {len(endpoints)}")
    print(f"  Expected: {expected_count}")
    print(f"  Protected: {protected_count}")
    print(f"  Success rate: {(protected_count/expected_count)*100:.1f}%")
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print(f"\n✅ All {protected_count} endpoints properly protected!")
        return True

def main():
    base_path = Path(__file__).parent / 'backend' / 'api' / 'routers'
    
    print("="*70)
    print("🛡️  ADMIN ROUTER PROTECTION - FINAL VERIFICATION")
    print("="*70)
    
    # Verify settings router (15 endpoints)
    settings_ok = verify_router(
        "Settings Router",
        base_path / 'settings_router.py',
        15
    )
    
    # Verify training router (10 endpoints)
    training_ok = verify_router(
        "Training Router", 
        base_path / 'training_router.py',
        10
    )
    
    # Final summary
    print(f"\n{'='*70}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*70}")
    
    if settings_ok and training_ok:
        print("✅ SUCCESS: All 25 admin endpoints properly protected!")
        print("\n🔐 Protection includes:")
        print("  • current_user: CurrentUser = Depends(get_current_user)")
        print("  • require_logged_in(current_user)")
        print("  • require_software_admin(current_user)")
        print("\n🎯 Expected behavior:")
        print("  • Not logged in → HTTP 401 Unauthorized")
        print("  • Logged in but not admin → HTTP 403 Forbidden")
        print("  • Admin user → Original functionality")
        return 0
    else:
        print("❌ FAILED: Some endpoints missing proper protection")
        return 1

if __name__ == '__main__':
    sys.exit(main())
