"""Test seasonal report ID stability on regeneration"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
from api.db_layer.seasonal_report import resolve_season_id_from_year_trimester

print("=" * 80)
print("TEST: Seasonal Report ID Stability on Regeneration")
print("=" * 80)

try:
    # Resolve season ID for 2026 Q1
    season_id = resolve_season_id_from_year_trimester(year=2026, trimester="Q1")
    print(f"\n✅ Season ID resolved: {season_id}")
    
    # Generate report FIRST TIME
    print(f"\n🔄 Generating seasonal report (FIRST TIME)...")
    report1 = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=1,
        user_id=1
    )
    
    report_id_1 = report1.get('header', {}).get('seasonal_report_id') if 'header' in report1 else report1.get('seasonal_report_id')
    print(f"✅ First generation - Report ID: {report_id_1}")
    print(f"   Full report keys: {list(report1.keys())}")
    
    # Generate report SECOND TIME (should UPDATE, not create new)
    print(f"\n🔄 Generating seasonal report (SECOND TIME - should UPDATE)...")
    report2 = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=1,
        user_id=1
    )
    
    report_id_2 = report2.get('header', {}).get('seasonal_report_id') if 'header' in report2 else report2.get('seasonal_report_id')
    print(f"✅ Second generation - Report ID: {report_id_2}")
    
    # Verify IDs match
    print(f"\n{'='*80}")
    if report_id_1 == report_id_2:
        print(f"✅ SUCCESS: Report ID remained stable!")
        print(f"   Both generations used the same ID: {report_id_1}")
        print(f"   This means action items are preserved! 🎉")
    else:
        print(f"❌ FAILURE: Report ID changed!")
        print(f"   First: {report_id_1}")
        print(f"   Second: {report_id_2}")
        print(f"   Action items would be lost!")
    print(f"{'='*80}")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
