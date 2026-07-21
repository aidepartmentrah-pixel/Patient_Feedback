"""
Debug both issues:
1. Force close subcase — test if DB update actually works
2. Seasonal report 924 — reproduce the 500 error
"""
import traceback

# ==========================================
# TEST 1: Seasonal report 924 — reproduce 500
# ==========================================
print("=== TEST 1: Seasonal Report 924 ===")
try:
    from api.db_layer.seasonal_report import (
        get_seasonal_report_keys_by_id,
        get_full_seasonal_report,
    )
    
    keys = get_seasonal_report_keys_by_id(924)
    print(f"Keys: {keys}")
    
    if keys:
        report = get_full_seasonal_report(
            season_id=keys['season_id'],
            orgunit_id=keys['orgunit_id'],
            orgunit_type=keys['orgunit_type']
        )
        if report:
            print(f"Report loaded OK. Header keys: {list(report['header'].keys())}")
            print(f"Classification stats: {len(report['classification_stats'])} rows")
            print(f"Policy snapshot: {report['policy_snapshot']}")
            
            # Try JSON serialization (what FastAPI does)
            import json
            json_str = json.dumps(report, default=str)
            print(f"JSON serialization OK ({len(json_str)} bytes)")
        else:
            print("ERROR: get_full_seasonal_report returned None")
    else:
        print("ERROR: keys is None")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# ==========================================
# TEST 2: Scope resolution — test if it works
# ==========================================
print("\n=== TEST 2: Scope Resolution ===")
try:
    from api.services.scope_resolver import resolve_user_scope
    
    # Check what resolve_user_scope expects
    import inspect
    sig = inspect.signature(resolve_user_scope)
    print(f"resolve_user_scope signature: {sig}")
    
    # Check the source briefly
    source = inspect.getsource(resolve_user_scope)
    # Print first 20 lines
    lines = source.split('\n')[:30]
    for line in lines:
        print(f"  {line}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()

# ==========================================
# TEST 3: Force close — check what happens
# ==========================================
print("\n=== TEST 3: Force Close DB Update ===")
try:
    from core.database import get_connection
    
    # Check subcase 525 BEFORE
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Status, ForceClosedAt FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = 525")
    r = cursor.fetchone()
    print(f"Subcase 525 BEFORE: Status={r[0]}, ForceClosedAt={r[1]}")
    cursor.close()
    conn.close()
    
    # Now call the actual DB function
    from api_v2.db_layer.administrative_subcase_db import force_close_subcase_with_tracking
    result = force_close_subcase_with_tracking(
        subcase_id=525,
        force_closed_by_user_id=1,  # admin user
        force_close_reason="Testing force close from debug script"
    )
    print(f"force_close_subcase_with_tracking returned: {result}")
    
    # Check AFTER
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT Status, ForceClosedAt, ForceCloseReason FROM dbo.APP_AdministrativeSubcase WHERE SubcaseID = 525")
    r = cursor.fetchone()
    print(f"Subcase 525 AFTER: Status={r[0]}, ForceClosedAt={r[1]}, Reason={r[2]}")
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
