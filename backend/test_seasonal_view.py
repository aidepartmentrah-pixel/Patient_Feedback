"""
Quick test: Verify the seasonal report detail flow works end-to-end.
Simulates what the endpoint does without HTTP.
"""
from core.database import get_connection
from api.db_layer.seasonal_report import (
    get_seasonal_report_keys_by_id,
    get_full_seasonal_report,
)

# Find a seasonal report ID that has a subcase for unit 95
conn = get_connection()
cursor = conn.cursor()
cursor.execute(
    "SELECT SubcaseID, SeasonalReportID, TargetOrgUnitID, Status "
    "FROM dbo.APP_AdministrativeSubcase "
    "WHERE TargetOrgUnitID = 95 AND CaseType = 'SEASONAL_REPORT_RESPONSE' "
    "AND Status = 'SUBMITTED_TO_SECTION'"
)
rows = cursor.fetchall()
print(f"Found {len(rows)} seasonal subcases for unit 95:")
for r in rows:
    print(f"  SubcaseID={r[0]}, ReportID={r[1]}, Status={r[3]}")
cursor.close()
conn.close()

if rows:
    report_id = rows[0][1]
    print(f"\nTesting with SeasonalReportID = {report_id}")
    
    # Step 1: Get keys
    keys = get_seasonal_report_keys_by_id(report_id)
    print(f"Keys: {keys}")
    
    if keys:
        # Step 2: Get full report
        report = get_full_seasonal_report(
            season_id=keys['season_id'],
            orgunit_id=keys['orgunit_id'],
            orgunit_type=keys['orgunit_type']
        )
        
        if report:
            h = report['header']
            print(f"\n=== REPORT HEADER ===")
            print(f"Period: {h.get('period')}")
            print(f"Org Unit: {h.get('orgunit_name')} (ID: {h.get('orgunit_id')})")
            print(f"Total Cases: {h.get('total_cases')}")
            print(f"Severity: Low={h.get('low_severity_count')}, Med={h.get('medium_severity_count')}, High={h.get('high_severity_count')}")
            print(f"Domains: Clinical={h.get('clinical_domain_count')}, Mgmt={h.get('management_domain_count')}, Rel={h.get('relational_domain_count')}")
            print(f"Compliant: {h.get('is_compliant')}")
            print(f"Violated Rules: {h.get('violated_rules')}")
            
            print(f"\n=== CLASSIFICATION STATS ({len(report['classification_stats'])} rows) ===")
            for s in report['classification_stats']:
                print(f"  {s.get('domain_name')} > {s.get('category_name')} > {s.get('classification_name')}: total={s.get('total_count')}")
            
            print(f"\n=== POLICY SNAPSHOT ===")
            print(f"  {report.get('policy_snapshot')}")
        else:
            print("ERROR: get_full_seasonal_report returned None")
    else:
        print("ERROR: get_seasonal_report_keys_by_id returned None")
else:
    print("No seasonal subcases found for unit 95")
