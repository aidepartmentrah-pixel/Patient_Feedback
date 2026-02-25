"""Test seasonal report generation"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
from api.db_layer.seasonal_report import resolve_season_id_from_year_trimester

print("=" * 80)
print("TEST: Seasonal Report Generation for 2026 Q1")
print("=" * 80)

try:
    # Resolve season ID for 2026 Q1
    season_id = resolve_season_id_from_year_trimester(year=2026, trimester="Q1")
    print(f"\n✅ Season ID resolved: {season_id}")
    
    # Generate report for orgunit 1, type 1 (example)
    print(f"\n🔄 Generating seasonal report...")
    print(f"   - Season: 2026 Q1 (ID={season_id})")
    print(f"   - OrgUnit: 1")
    print(f"   - OrgUnit Type: 1")
    
    report = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=1,
        user_id=1
    )
    
    print(f"\n✅ Report generated successfully!")
    print(f"\nReport Summary:")
    if 'seasonal_report_id' in report:
        print(f"   - Report ID: {report['seasonal_report_id']}")
    if 'total_cases' in report:
        print(f"   - Total Cases: {report['total_cases']}")
    if 'is_compliant' in report:
        print(f"   - Compliant: {report['is_compliant']}")
    
    print(f"\n✅ SUCCESS: Seasonal report generation works!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("=" * 80)
