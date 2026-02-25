"""
Test Seasonal Multi-Export Implementation
Tests hospital-level aggregation and multi-export ZIP generation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.api.db_layer.seasonal_report_aggregation import (
    get_seasonal_classification_stats,
    get_seasonal_domain_totals
)

print("\n" + "="*100)
print("TESTING SEASONAL MULTI-EXPORT IMPLEMENTATION")
print("="*100 + "\n")

# Test data from diagnostic:
# - Season ID 5 = Q1-2026 (2026-01-01 to 2026-03-31)
# - 10 total cases in this period
# - Cases distributed across OrgUnits: 43 (2), 183 (2), 25 (1), 42 (1), 44 (1), 45 (1), 46 (1), 136 (1)

season_id = 5  # Q1-2026
year = 2026
period = "Q1"

print(f"Test Scenario: {period} {year} (season_id={season_id})")
print(f"Expected: 10 total cases across multiple OrgUnits\n")

# ============================================================================
# TEST 1: Hospital-Level Report (orgunit_type=0)
# Expected: Aggregate ALL 10 cases
# ============================================================================

print("="*80)
print("TEST 1: Hospital-Level Report (orgunit_id=1, orgunit_type=0)")
print("="*80)

try:
    classification_stats = get_seasonal_classification_stats(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=0
    )
    
    domain_totals = get_seasonal_domain_totals(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=0
    )
    
    print(f"✅ Hospital-Level Aggregation Success!")
    print(f"\nDomain Totals:")
    print(f"  Total Cases: {domain_totals['total_cases']}")
    print(f"  Clinical: {domain_totals['clinical_domain_count']}")
    print(f"  Management: {domain_totals['management_domain_count']}")
    print(f"  Relational: {domain_totals['relational_domain_count']}")
    print(f"  Low Severity: {domain_totals['low_severity_count']}")
    print(f"  Medium Severity: {domain_totals['medium_severity_count']}")
    print(f"  High Severity: {domain_totals['high_severity_count']}")
    
    print(f"\nClassification Stats: {len(classification_stats)} classifications")
    for stat in classification_stats[:5]:  # Show first 5
        print(f"  Classification {stat['classification_id']}: {stat['total_count']} cases")
    
    if domain_totals['total_cases'] == 10:
        print(f"\n✅ PASS: Expected 10 cases, got {domain_totals['total_cases']}")
    else:
        print(f"\n❌ FAIL: Expected 10 cases, got {domain_totals['total_cases']}")

except Exception as e:
    print(f"❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n")

# ============================================================================
# TEST 2: Specific Administration Report (orgunit_id=43, orgunit_type=1)
# Expected: 2 cases for OrgUnit 43
# ============================================================================

print("="*80)
print("TEST 2: Specific Administration Report (orgunit_id=43, orgunit_type=1)")
print("="*80)

try:
    classification_stats = get_seasonal_classification_stats(
        season_id=season_id,
        orgunit_id=43,
        orgunit_type=1
    )
    
    domain_totals = get_seasonal_domain_totals(
        season_id=season_id,
        orgunit_id=43,
        orgunit_type=1
    )
    
    print(f"✅ Specific Unit Aggregation Success!")
    print(f"\nDomain Totals for OrgUnit 43:")
    print(f"  Total Cases: {domain_totals['total_cases']}")
    print(f"  Clinical: {domain_totals['clinical_domain_count']}")
    print(f"  Management: {domain_totals['management_domain_count']}")
    print(f"  Relational: {domain_totals['relational_domain_count']}")
    
    if domain_totals['total_cases'] == 2:
        print(f"\n✅ PASS: Expected 2 cases for OrgUnit 43, got {domain_totals['total_cases']}")
    else:
        print(f"\n⚠️ WARNING: Expected 2 cases for OrgUnit 43, got {domain_totals['total_cases']}")

except Exception as e:
    print(f"❌ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n")

# ============================================================================
# TEST 3: Multi-Export Service Check (Import Test)
# ============================================================================

print("="*80)
print("TEST 3: Multi-Seasonal Export Service - Import Test")
print("="*80)

try:
    from backend.api.services.multi_seasonal_export_service import multi_seasonal_export_service
    print("✅ multi_seasonal_export_service imported successfully")
    
    # Check methods exist
    if hasattr(multi_seasonal_export_service, 'generate_multi_seasonal_export'):
        print("✅ generate_multi_seasonal_export() method exists")
    else:
        print("❌ generate_multi_seasonal_export() method NOT FOUND")
    
    if hasattr(multi_seasonal_export_service, '_generate_unit_seasonal_report'):
        print("✅ _generate_unit_seasonal_report() method exists")
    else:
        print("❌ _generate_unit_seasonal_report() method NOT FOUND")
        
except Exception as e:
    print(f"❌ IMPORT FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("="*100)
print("TESTS COMPLETE")
print("="*100)
print("\n✅ Next Steps:")
print("  1. Restart uvicorn server: uvicorn backend.main:app --reload")
print("  2. Test hospital-level view in frontend (All Hospital button)")
print("  3. Test multi-export ZIP (All Administrations button + Export)")
print("  4. Verify ZIP contains multiple files for each OrgUnit with data")
print("\n")
