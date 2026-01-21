"""
Test Sanity Check for Seasonal Report Generation
Verifies that TotalCases = Sum of Domain Counts
"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
from api.db_layer.seasonal_report import resolve_season_id_from_year_trimester

print("\n" + "="*100)
print("SANITY CHECK: Seasonal Report Data Integrity Test")
print("="*100 + "\n")

try:
    # Test with Q1 2026
    season_id = resolve_season_id_from_year_trimester(year=2026, trimester="Q1")
    print(f"✅ Season ID resolved: {season_id}\n")
    
    # Generate seasonal report for Hospital level
    print("🔄 Generating seasonal report for Hospital level...")
    report = get_or_generate_seasonal_report(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=0,  # Hospital
        user_id=1
    )
    
    # Extract data
    header = report.get('header', {})
    total_cases = header.get('total_cases', 0)
    clinical = header.get('clinical_domain_count', 0)
    management = header.get('management_domain_count', 0)
    relational = header.get('relational_domain_count', 0)
    
    low_sev = header.get('low_severity_count', 0)
    med_sev = header.get('medium_severity_count', 0)
    high_sev = header.get('high_severity_count', 0)
    
    domain_sum = clinical + management + relational
    severity_sum = low_sev + med_sev + high_sev
    
    # Display results
    print("\n" + "="*100)
    print("📊 REPORT DATA SUMMARY")
    print("="*100)
    print(f"Total Cases:         {total_cases}")
    print(f"\nDomain Breakdown:")
    print(f"  Clinical:          {clinical}")
    print(f"  Management:        {management}")
    print(f"  Relational:        {relational}")
    print(f"  Domain Sum:        {domain_sum}")
    print(f"\nSeverity Breakdown:")
    print(f"  Low:               {low_sev}")
    print(f"  Medium:            {med_sev}")
    print(f"  High:              {high_sev}")
    print(f"  Severity Sum:      {severity_sum}")
    
    # Sanity checks
    print("\n" + "="*100)
    print("✅ SANITY CHECK RESULTS")
    print("="*100)
    
    domain_check = "✅ PASS" if domain_sum == total_cases else f"❌ FAIL (Expected {total_cases}, got {domain_sum})"
    severity_check = "✅ PASS" if severity_sum == total_cases else f"❌ FAIL (Expected {total_cases}, got {severity_sum})"
    
    print(f"Domain Sum = Total Cases:    {domain_check}")
    print(f"Severity Sum = Total Cases:  {severity_check}")
    
    # Check percentages
    if total_cases > 0:
        print("\n" + "="*100)
        print("📈 DOMAIN PERCENTAGES")
        print("="*100)
        clinical_pct = (clinical / total_cases) * 100
        management_pct = (management / total_cases) * 100
        relational_pct = (relational / total_cases) * 100
        
        print(f"Clinical:   {clinical_pct:.1f}%")
        print(f"Management: {management_pct:.1f}%")
        print(f"Relational: {relational_pct:.1f}%")
        print(f"Total:      {clinical_pct + management_pct + relational_pct:.1f}%")
        
        # Check if any percentage > 100%
        if any(pct > 100 for pct in [clinical_pct, management_pct, relational_pct]):
            print("\n❌ ERROR: One or more percentages exceed 100%!")
        else:
            print("\n✅ All percentages are valid (≤100%)")
    
    # Check violated rules
    violated_rules = header.get('violated_rules')
    if violated_rules:
        import json
        print("\n" + "="*100)
        print("⚠️  VIOLATED RULES")
        print("="*100)
        rules = json.loads(violated_rules)
        for rule in rules:
            print(f"\nRule: {rule['rule']}")
            print(f"  Name (EN): {rule.get('rule_name_en', 'N/A')}")
            print(f"  Name (AR): {rule.get('rule_name_ar', 'N/A')}")
            print(f"  Threshold: {rule['threshold']}{rule.get('threshold_unit', '')}")
            print(f"  Actual: {rule['actual']}{rule.get('actual_unit', '')}")
            
            # Check if percentage is sane
            if rule.get('threshold_unit') == '%' and rule['actual'] > 100:
                print(f"  ❌ ERROR: Percentage {rule['actual']}% exceeds 100%!")
    else:
        print("\n✅ No policy violations detected")
    
    print("\n" + "="*100)
    print("TEST COMPLETE")
    print("="*100 + "\n")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
