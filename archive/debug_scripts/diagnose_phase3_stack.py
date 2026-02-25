"""
Phase 3 Stack Diagnostic Tool
Scans the entire stack from Database → DB Layer → Service Layer
to identify what's working and what's broken.
"""

import sys
import os

# Force UTF-8 encoding
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add backend to path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection

print("\n" + "="*80)
print("PHASE 3 STACK DIAGNOSTIC TOOL")
print("="*80)

results = {
    'database': {},
    'db_layer': {},
    'service': {},
    'adapter': {}
}

# =============================================================================
# LEVEL 1: DATABASE TABLES
# =============================================================================
print("\n" + "="*80)
print("LEVEL 1: DATABASE TABLES")
print("="*80)

conn = get_connection()
cursor = conn.cursor()

# Check each required table
tables_to_check = [
    # STEP 3.1 - Lookup tables
    ('APP_LOOKUP_CaseType', 'STEP 3.1'),
    ('APP_LOOKUP_CaseStatus', 'STEP 3.1'),
    ('APP_LOOKUP_AssignmentRole', 'STEP 3.1'),
    ('APP_LOOKUP_PriorityLevel', 'STEP 3.1'),
    ('APP_LOOKUP_ActionItemType', 'STEP 3.1'),
    
    # STEP 3.2 - Administrative Subcase
    ('APP_AdministrativeSubcase', 'STEP 3.2'),
    
    # STEP 3.3 - Action Item Subcase
    ('APP_ActionItemSubcase', 'STEP 3.3'),
]

for table_name, step in tables_to_check:
    try:
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table_name}")
        row = cursor.fetchone()
        count = row.cnt if row else 0
        results['database'][table_name] = {'exists': True, 'count': count, 'step': step}
        print(f"✅ {table_name:40} | Count: {count:5} | {step}")
    except Exception as e:
        results['database'][table_name] = {'exists': False, 'error': str(e), 'step': step}
        print(f"❌ {table_name:40} | NOT FOUND | {step}")

# Check the existing subcase
print("\n[DATABASE] Checking existing SubcaseID 53...")
try:
    cursor.execute("""
        SELECT 
            SubcaseID,
            CaseType,
            IncidentRequestCaseID,
            SeasonalReportID,
            TargetOrgUnitID,
            Status
        FROM APP_AdministrativeSubcase
        WHERE SubcaseID = 53
    """)
    row = cursor.fetchone()
    if row:
        print(f"  ✅ SubcaseID 53 found:")
        print(f"     CaseType: {row.CaseType}")
        print(f"     IncidentRequestCaseID: {row.IncidentRequestCaseID}")
        print(f"     SeasonalReportID: {row.SeasonalReportID}")
        print(f"     TargetOrgUnitID: {row.TargetOrgUnitID}")
        print(f"     Status: {row.Status}")
    else:
        print(f"  ❌ SubcaseID 53 not found")
except Exception as e:
    print(f"  ❌ Error: {str(e)}")

cursor.close()
conn.close()

# =============================================================================
# LEVEL 2: DB LAYER FUNCTIONS
# =============================================================================
print("\n" + "="*80)
print("LEVEL 2: DB LAYER FUNCTIONS")
print("="*80)

# STEP 3.6 - administrative_subcase_db.py
print("\n[STEP 3.6] Testing administrative_subcase_db.py...")
try:
    from api_v2.db_layer import administrative_subcase_db
    results['db_layer']['administrative_subcase_db'] = {'imported': True}
    print("  ✅ Module imported successfully")
    
    # Check functions exist
    functions_to_check = [
        'create_subcase',
        'get_subcase_by_id',
        'get_subcases_by_incident',
        'get_subcases_by_seasonal_report',
        'update_subcase_status',
        'assign_subcase_to_role',
        'add_note_to_subcase',
    ]
    
    for func_name in functions_to_check:
        if hasattr(administrative_subcase_db, func_name):
            print(f"    ✅ {func_name}")
            results['db_layer'][f'administrative_subcase_db.{func_name}'] = {'exists': True}
        else:
            print(f"    ❌ {func_name} NOT FOUND")
            results['db_layer'][f'administrative_subcase_db.{func_name}'] = {'exists': False}
    
    # Test actual DB operation - get existing subcase 53
    print("\n  [TEST] Getting SubcaseID 53...")
    try:
        subcase = administrative_subcase_db.get_subcase_by_id(53)
        if subcase:
            print(f"    ✅ get_subcase_by_id(53) works!")
            print(f"       Returned: {subcase.get('case_type')}")
            results['db_layer']['get_subcase_by_id'] = {'works': True}
        else:
            print(f"    ⚠️  get_subcase_by_id(53) returned None")
            results['db_layer']['get_subcase_by_id'] = {'works': False, 'reason': 'returned None'}
    except Exception as e:
        print(f"    ❌ get_subcase_by_id(53) failed: {str(e)}")
        results['db_layer']['get_subcase_by_id'] = {'works': False, 'error': str(e)}
    
except ImportError as e:
    print(f"  ❌ Import failed: {str(e)}")
    results['db_layer']['administrative_subcase_db'] = {'imported': False, 'error': str(e)}

# STEP 3.7 - action_item_subcase_db.py
print("\n[STEP 3.7] Testing action_item_subcase_db.py...")
try:
    from api_v2.db_layer import action_item_subcase_db
    results['db_layer']['action_item_subcase_db'] = {'imported': True}
    print("  ✅ Module imported successfully")
    
    # Check functions exist
    functions_to_check = [
        'create_action_item',
        'get_action_items_by_subcase',
        'update_action_item_status',
    ]
    
    for func_name in functions_to_check:
        if hasattr(action_item_subcase_db, func_name):
            print(f"    ✅ {func_name}")
            results['db_layer'][f'action_item_subcase_db.{func_name}'] = {'exists': True}
        else:
            print(f"    ❌ {func_name} NOT FOUND")
            results['db_layer'][f'action_item_subcase_db.{func_name}'] = {'exists': False}
    
except ImportError as e:
    print(f"  ❌ Import failed: {str(e)}")
    results['db_layer']['action_item_subcase_db'] = {'imported': False, 'error': str(e)}

# =============================================================================
# LEVEL 3: SERVICE LAYER FUNCTIONS
# =============================================================================
print("\n" + "="*80)
print("LEVEL 3: SERVICE LAYER FUNCTIONS")
print("="*80)

# STEP 3.9 - case_creation_service.py
print("\n[STEP 3.9] Testing case_creation_service.py...")
try:
    from api_v2.services import case_creation_service
    results['service']['case_creation_service'] = {'imported': True}
    print("  ✅ Module imported successfully")
    
    # Check functions exist
    functions_to_check = [
        'create_subcases_for_incident',
        'create_subcases_for_seasonal_report',
    ]
    
    for func_name in functions_to_check:
        if hasattr(case_creation_service, func_name):
            print(f"    ✅ {func_name}")
            results['service'][f'case_creation_service.{func_name}'] = {'exists': True}
        else:
            print(f"    ❌ {func_name} NOT FOUND")
            results['service'][f'case_creation_service.{func_name}'] = {'exists': False}
    
except ImportError as e:
    print(f"  ❌ Import failed: {str(e)}")
    results['service']['case_creation_service'] = {'imported': False, 'error': str(e)}

# =============================================================================
# LEVEL 4: ADAPTER HOOKS
# =============================================================================
print("\n" + "="*80)
print("LEVEL 4: ADAPTER HOOKS")
print("="*80)

# Check insert_service.py
print("\n[STEP 3.10] Checking insert_service.py adapter...")
try:
    with open('backend/api/services/insert_service.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'import': 'from backend.api_v2.services.case_creation_service import create_subcases_for_incident' in content,
        'call': 'create_subcases_for_incident(new_id' in content,
        'try_except': 'try:' in content and 'API V2 ADAPTER' in content,
    }
    
    for check_name, passed in checks.items():
        if passed:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
    
    results['adapter']['insert_service'] = checks
    
except Exception as e:
    print(f"  ❌ Error reading file: {str(e)}")
    results['adapter']['insert_service'] = {'error': str(e)}

# Check seasonal_report_generator.py
print("\n[STEP 3.10] Checking seasonal_report_generator.py adapter...")
try:
    with open('backend/api/services/seasonal_report_generator.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    checks = {
        'import': 'from backend.api_v2.services.case_creation_service import create_subcases_for_seasonal_report' in content,
        'call': 'create_subcases_for_seasonal_report(seasonal_report_id' in content,
        'try_except': 'try:' in content and 'API V2 ADAPTER' in content,
    }
    
    for check_name, passed in checks.items():
        if passed:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name}")
    
    results['adapter']['seasonal_report_generator'] = checks
    
except Exception as e:
    print(f"  ❌ Error reading file: {str(e)}")
    results['adapter']['seasonal_report_generator'] = {'error': str(e)}

# =============================================================================
# SUMMARY & DIAGNOSIS
# =============================================================================
print("\n" + "="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)

# Count issues by level
database_issues = sum(1 for v in results['database'].values() if not v.get('exists', False))
db_layer_issues = sum(1 for k, v in results['db_layer'].items() if not v.get('imported', True) or not v.get('exists', True) or not v.get('works', True))
service_issues = sum(1 for k, v in results['service'].items() if not v.get('imported', True) or not v.get('exists', True))
adapter_issues = sum(1 for v in results['adapter'].values() if isinstance(v, dict) and not all(v.values()))

print(f"\nLevel 1 - Database Tables:  {len(results['database']) - database_issues}/{len(results['database'])} OK")
if database_issues > 0:
    print(f"  ❌ {database_issues} table(s) missing")
    for table, info in results['database'].items():
        if not info.get('exists', False):
            print(f"     - {table} ({info['step']})")

print(f"\nLevel 2 - DB Layer:         {len(results['db_layer']) - db_layer_issues}/{len(results['db_layer'])} OK")
if db_layer_issues > 0:
    print(f"  ❌ {db_layer_issues} issue(s)")
    for func, info in results['db_layer'].items():
        if not info.get('imported', True) or not info.get('exists', True) or not info.get('works', True):
            print(f"     - {func}")

print(f"\nLevel 3 - Service Layer:    {len(results['service']) - service_issues}/{len(results['service'])} OK")
if service_issues > 0:
    print(f"  ❌ {service_issues} issue(s)")
    for func, info in results['service'].items():
        if not info.get('imported', True) or not info.get('exists', True):
            print(f"     - {func}")

print(f"\nLevel 4 - Adapter Hooks:    {2 - adapter_issues}/2 OK")
if adapter_issues > 0:
    print(f"  ❌ {adapter_issues} issue(s)")
    for adapter, checks in results['adapter'].items():
        if isinstance(checks, dict) and not all(checks.values()):
            print(f"     - {adapter}")

# =============================================================================
# RECOMMENDED ACTIONS
# =============================================================================
print("\n" + "="*80)
print("RECOMMENDED ACTIONS")
print("="*80)

if database_issues > 0:
    print("\n🔴 CRITICAL: Database tables missing")
    print("   Action: Run Phase 3 database migration scripts")
    print("   Files: backend/database_migrations/phase3_step*.sql")
    
if db_layer_issues > 0:
    print("\n🟡 WARNING: DB Layer issues detected")
    print("   Action: Check/fix db_layer files")
    print("   Files: backend/api_v2/db_layer/*.py")

if service_issues > 0:
    print("\n🟡 WARNING: Service layer issues detected")
    print("   Action: Check/fix service files")
    print("   Files: backend/api_v2/services/*.py")

if adapter_issues > 0:
    print("\n🟡 WARNING: Adapter hooks incomplete")
    print("   Action: Re-run STEP 3.10 adapter installation")

if database_issues == 0 and db_layer_issues == 0 and service_issues == 0 and adapter_issues == 0:
    print("\n✅ ALL SYSTEMS OPERATIONAL!")
    print("   The entire Phase 3 stack is working correctly.")
    print("   Ready to proceed with testing.")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
