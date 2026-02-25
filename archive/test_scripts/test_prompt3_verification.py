"""
Prompt 3 - Protect Explanations + PARTIAL Reports Router Verification
=========================================================================
Verifies that:
- ALL explanation_routes.py endpoints have authentication guards
- admin_force_close_case_endpoint has BOTH authentication + admin authorization
- ONLY specified reports_router.py endpoints have guards
- Public endpoints (view_seasonal_report, view_monthly_report, download_export) remain unchanged
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
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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

def verify_explanation_routes():
    """Verify ALL endpoints in explanation_routes.py have guards."""
    print(f"\n{'='*70}")
    print(f"🔍 Part A: Verifying explanation_routes.py")
    print(f"{'='*70}")
    
    file_path = Path(__file__).parent / 'backend' / 'api' / 'routers' / 'explanation_routes.py'
    functions = extract_function_info(file_path)
    
    # All endpoint functions (exclude internal helpers)
    all_endpoints = [
        'get_pending_explanations_endpoint',
        'get_explanation_statistics',
        'get_case_explanation_details_endpoint',
        'get_case_completion_status_endpoint',
        'submit_explanation_endpoint',
        'update_requires_explanation_flag',
        'admin_force_close_case_endpoint',  # Special: needs admin guard too
        'check_case_for_automatic_closure',
        'mark_action_item_complete_endpoint',
        'validate_explanation_endpoint'
    ]
    
    protected_count = 0
    issues = []
    
    for name in all_endpoints:
        if name not in functions:
            issues.append(f"  ❌ {name}: Function not found")
            continue
        
        info = functions[name]
        
        # Special case: admin_force_close_case_endpoint needs BOTH guards
        if name == 'admin_force_close_case_endpoint':
            if info['has_current_user'] and info['has_require_logged_in'] and info['has_require_admin']:
                protected_count += 1
                print(f"  ✅ {name} (line {info['line']}) - AUTH + ADMIN ⭐")
            else:
                missing = []
                if not info['has_current_user']:
                    missing.append('current_user parameter')
                if not info['has_require_logged_in']:
                    missing.append('require_logged_in()')
                if not info['has_require_admin']:
                    missing.append('require_software_admin()')
                issues.append(f"  ❌ {name} (line {info['line']}): Missing {', '.join(missing)}")
        else:
            # All other endpoints: just need require_logged_in
            if info['has_current_user'] and info['has_require_logged_in']:
                protected_count += 1
                print(f"  ✅ {name} (line {info['line']}) - AUTH")
            else:
                missing = []
                if not info['has_current_user']:
                    missing.append('current_user parameter')
                if not info['has_require_logged_in']:
                    missing.append('require_logged_in()')
                issues.append(f"  ❌ {name} (line {info['line']}): Missing {', '.join(missing)}")
    
    print(f"\n📊 Results:")
    print(f"  Total endpoints: {len(all_endpoints)}")
    print(f"  Protected: {protected_count}")
    print(f"  Success rate: {(protected_count/len(all_endpoints))*100:.1f}%")
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print(f"\n✅ All {protected_count} endpoints properly protected!")
        return True

def verify_reports_router():
    """Verify ONLY specified endpoints in reports_router.py have guards."""
    print(f"\n{'='*70}")
    print(f"🔍 Part B: Verifying reports_router.py (PARTIAL PROTECTION)")
    print(f"{'='*70}")
    
    file_path = Path(__file__).parent / 'backend' / 'api' / 'routers' / 'reports_router.py'
    functions = extract_function_info(file_path)
    
    # Endpoints that MUST be protected
    protected_endpoints = [
        'submit_explanation',
        'update_explanation',
        'export_report',
        'export_seasonal_report',
        'export_monthly_report'
    ]
    
    # Endpoints that MUST remain public (no guards)
    public_endpoints = [
        'view_seasonal_report',
        'view_monthly_report',
        'download_export'
    ]
    
    protected_count = 0
    public_count = 0
    issues = []
    
    print("\n🔒 Protected Endpoints (must have guards):")
    for name in protected_endpoints:
        if name not in functions:
            issues.append(f"  ❌ {name}: Function not found")
            continue
        
        info = functions[name]
        if info['has_current_user'] and info['has_require_logged_in']:
            protected_count += 1
            print(f"  ✅ {name} (line {info['line']}) - AUTH")
        else:
            missing = []
            if not info['has_current_user']:
                missing.append('current_user parameter')
            if not info['has_require_logged_in']:
                missing.append('require_logged_in()')
            issues.append(f"  ❌ {name} (line {info['line']}): Missing {', '.join(missing)}")
    
    print("\n🌐 Public Endpoints (must NOT have guards):")
    for name in public_endpoints:
        if name not in functions:
            issues.append(f"  ❌ {name}: Function not found")
            continue
        
        info = functions[name]
        if not info['has_current_user'] and not info['has_require_logged_in']:
            public_count += 1
            print(f"  ✅ {name} (line {info['line']}) - PUBLIC (no guards)")
        else:
            problems = []
            if info['has_current_user']:
                problems.append('has current_user parameter')
            if info['has_require_logged_in']:
                problems.append('has require_logged_in()')
            issues.append(f"  ❌ {name} (line {info['line']}): Should be PUBLIC but {', '.join(problems)}")
    
    print(f"\n📊 Results:")
    print(f"  Protected endpoints: {protected_count}/{len(protected_endpoints)}")
    print(f"  Public endpoints: {public_count}/{len(public_endpoints)}")
    print(f"  Protected success rate: {(protected_count/len(protected_endpoints))*100:.1f}%")
    print(f"  Public success rate: {(public_count/len(public_endpoints))*100:.1f}%")
    
    if issues:
        print(f"\n❌ Issues found:")
        for issue in issues:
            print(issue)
        return False
    else:
        print(f"\n✅ All endpoints correctly configured!")
        print(f"  - {protected_count} protected with authentication")
        print(f"  - {public_count} remain public")
        return True

def main():
    print("="*70)
    print("🛡️  PROMPT 3 - EXPLANATIONS + PARTIAL REPORTS VERIFICATION")
    print("="*70)
    
    # Verify Part A: explanation_routes.py (ALL endpoints)
    part_a_ok = verify_explanation_routes()
    
    # Verify Part B: reports_router.py (PARTIAL protection)
    part_b_ok = verify_reports_router()
    
    # Final summary
    print(f"\n{'='*70}")
    print("📋 FINAL SUMMARY")
    print(f"{'='*70}")
    
    if part_a_ok and part_b_ok:
        print("✅ SUCCESS: All requirements met!")
        print("\n🎯 Part A (explanation_routes.py):")
        print("  • All 10 endpoints have require_logged_in guard")
        print("  • admin_force_close_case_endpoint has require_software_admin guard too")
        print("\n🎯 Part B (reports_router.py):")
        print("  • 5 endpoints protected: submit_explanation, update_explanation,")
        print("    export_report, export_seasonal_report, export_monthly_report")
        print("  • 3 endpoints remain PUBLIC: view_seasonal_report,")
        print("    view_monthly_report, download_export")
        print("\n🔐 Expected behavior:")
        print("  • Protected endpoints → 401 if not logged in")
        print("  • admin_force_close_case_endpoint → 403 if not admin")
        print("  • Public endpoints → accessible without authentication")
        return 0
    else:
        print("❌ FAILED: Some requirements not met")
        if not part_a_ok:
            print("  ⚠️ Part A (explanation_routes.py) has issues")
        if not part_b_ok:
            print("  ⚠️ Part B (reports_router.py) has issues")
        return 1

if __name__ == '__main__':
    sys.exit(main())
