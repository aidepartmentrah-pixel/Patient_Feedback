"""
Test script to debug seasonal report freeze issue
Tests: year=2026, trimester=Q1, orgunit_id=1, orgunit_type=0, user_id=1
"""

import sys
import signal
from datetime import datetime

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")

# Set up signal for timeout (30 seconds)
if sys.platform != 'win32':
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)

print("="*80)
print("SEASONAL REPORT FREEZE DIAGNOSTIC TEST")
print("="*80)
print(f"Test Parameters:")
print(f"  - Year: 2026")
print(f"  - Period: Q1")
print(f"  - OrgUnit ID: 1")
print(f"  - OrgUnit Type: 0 (Hospital)")
print(f"  - User ID: 1")
print("="*80)
print()

try:
    # Step 1: Import backend modules
    print("[1/6] Importing backend modules...")
    start_time = datetime.now()
    
    sys.path.insert(0, 'backend')
    from backend.api.db_layer.seasonal_report import create_season_if_not_exists
    from backend.api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
    
    # Helper to resolve season
    def resolve_season_id_from_year_period(year, period):
        from backend.api.db_layer.seasonal_report import resolve_season_id_from_year_trimester
        # Try direct lookup first
        season_id = resolve_season_id_from_year_trimester(year, period)
        return season_id
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   SUCCESS - Imports successful ({elapsed:.2f}s)")
    print()
    
    # Step 2: Resolve season
    print("[2/6] Resolving season ID for 2026-Q1...")
    start_time = datetime.now()
    
    season_id = resolve_season_id_from_year_period(year=2026, period='Q1')
    if season_id is None:
        print("   ! Season not found, creating...")
        season_id = create_season_if_not_exists(year=2026, period='Q1')
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   SUCCESS - Season ID: {season_id} ({elapsed:.2f}s)")
    print()
    
    # Step 3: Generate report
    print("[3/6] Generating seasonal report...")
    print("   (This is where the freeze likely occurs)")
    start_time = datetime.now()
    
    report = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=0,
        user_id=1
    )
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   SUCCESS - Report generated ({elapsed:.2f}s)")
    print(f"   Report ID: {report.get('seasonal_report_id')}")
    print(f"   Total Cases: {report.get('header', {}).get('total_cases', 0)}")
    print(f"   Is Compliant: {report.get('header', {}).get('is_compliant')}")
    print()
    
    # Step 4: Test Word export
    print("[4/6] Testing Word document generation...")
    start_time = datetime.now()
    
    from backend.api.services.seasonal_report_formatter import generate_seasonal_word_report
    
    word_bytes = generate_seasonal_word_report(report, language='ar')
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"   SUCCESS - Word document generated ({elapsed:.2f}s)")
    print(f"   Document size: {len(word_bytes)} bytes")
    print()
    
    # Step 5: Save test file
    print("[5/6] Saving test output...")
    with open('test_seasonal_output.docx', 'wb') as f:
        f.write(word_bytes)
    print(f"   SUCCESS - Saved to: test_seasonal_output.docx")
    print()
    
    # Step 6: Summary
    print("[6/6] Test Summary")
    print("="*80)
    print("SUCCESS - ALL TESTS PASSED - No freeze detected!")
    print("="*80)
    
except TimeoutError:
    elapsed = (datetime.now() - start_time).total_seconds()
    print()
    print("="*80)
    print(f"TIMEOUT after {elapsed:.2f}s")
    print("="*80)
    print("The operation is hanging. Check the last output line to see where.")
    sys.exit(1)
    
except KeyboardInterrupt:
    print()
    print("="*80)
    print("INTERRUPTED by user")
    print("="*80)
    sys.exit(1)
    
except Exception as e:
    import traceback
    print()
    print("="*80)
    print(f"ERROR: {type(e).__name__}")
    print("="*80)
    print(f"{str(e)}")
    print()
    print("Traceback:")
    print("-"*80)
    traceback.print_exc()
    sys.exit(1)
