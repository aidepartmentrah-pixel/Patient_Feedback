"""
Test seasonal report generation for Administration units to verify data integrity fix.
This test verifies that Administration-level seasonal reports now retrieve data correctly
using target department filtering (same mechanism as monthly reporting).
"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from api.db_layer.seasonal_report_aggregation import (
    get_seasonal_classification_stats,
    get_seasonal_domain_totals
)
from api.db_layer.seasonal_report import resolve_season_id_from_year_trimester
from api.db_layer.admin_units import get_units_by_type

print("=" * 100)
print("TEST: Seasonal Report Administration Data Integrity Fix")
print("=" * 100)
print()

# Configuration
TEST_YEAR = 2026
TEST_PERIOD = "Q1"

try:
    # Resolve season ID
    season_id = resolve_season_id_from_year_trimester(year=TEST_YEAR, trimester=TEST_PERIOD)
    print(f"✅ Season ID resolved: {season_id} ({TEST_PERIOD} {TEST_YEAR})")
    print()
    
    # ========================================================================
    # TEST 1: Hospital Level (orgunit_id=1, orgunit_type=0)
    # Expected: Should show all complaints
    # ========================================================================
    print("=" * 100)
    print("TEST 1: Hospital Level (orgunit_id=1, orgunit_type=0)")
    print("=" * 100)
    
    hospital_totals = get_seasonal_domain_totals(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=0
    )
    
    print(f"Hospital Total Cases: {hospital_totals['total_cases']}")
    print(f"  - Clinical Domain: {hospital_totals['clinical_domain_count']}")
    print(f"  - Management Domain: {hospital_totals['management_domain_count']}")
    print(f"  - Relational Domain: {hospital_totals['relational_domain_count']}")
    print(f"  - Low Severity: {hospital_totals['low_severity_count']}")
    print(f"  - Medium Severity: {hospital_totals['medium_severity_count']}")
    print(f"  - High Severity: {hospital_totals['high_severity_count']}")
    print()
    
    # ========================================================================
    # TEST 2: Administration Level (orgunit_type=1)
    # Expected: Should now show data (FIXED)
    # ========================================================================
    print("=" * 100)
    print("TEST 2: Administration Level - First 9 Administrations (orgunit_type=1)")
    print("=" * 100)
    
    # Get all administrations (Type 323)
    administrations = get_units_by_type(323)
    print(f"Found {len(administrations)} administrations in database")
    print()
    
    admin_results = []
    total_admin_cases = 0
    
    # Test first 9 administrations
    for i, admin in enumerate(administrations[:9], 1):
        admin_id = admin["id"]
        admin_name = admin["name"]
        
        totals = get_seasonal_domain_totals(
            season_id=season_id,
            orgunit_id=admin_id,
            orgunit_type=1  # Administration
        )
        
        total_cases = totals['total_cases']
        total_admin_cases += total_cases
        
        admin_results.append({
            'id': admin_id,
            'name': admin_name,
            'count': total_cases
        })
        
        status = "✅" if total_cases > 0 else "⚠️"
        print(f"{status} Administration {i}: {admin_name}")
        print(f"   Total Cases: {total_cases}")
        if total_cases > 0:
            print(f"   - Clinical: {totals['clinical_domain_count']}, Management: {totals['management_domain_count']}, Relational: {totals['relational_domain_count']}")
        print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    
    print(f"\n📊 Hospital Level:")
    print(f"  Total Cases: {hospital_totals['total_cases']}")
    
    print(f"\n📊 Administration Level (9 units tested):")
    print(f"  Total Cases Across Administrations: {total_admin_cases}")
    
    non_empty_count = sum(1 for r in admin_results if r['count'] > 0)
    empty_count = sum(1 for r in admin_results if r['count'] == 0)
    
    print(f"  Non-Empty Administrations: {non_empty_count}")
    print(f"  Empty Administrations: {empty_count}")
    
    print("\n✅ TEST RESULT:")
    if non_empty_count > 0:
        print("  SUCCESS! Administration-level reports now retrieve data correctly.")
        print("  The target department filtering with tree expansion is working.")
    else:
        print("  ⚠️  WARNING: All administration reports still show 0 cases.")
        print("  This might indicate no data exists for this period, or another issue.")
    
    print()
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
