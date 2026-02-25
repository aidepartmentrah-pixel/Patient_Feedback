import sys
sys.path.insert(0, 'backend')

from api.db_layer.explanation_seasonal_db import get_seasonal_reports_needing_explanation

# Test the function
result = get_seasonal_reports_needing_explanation(non_compliant_only=True)

print("Success:", result['success'])
print("Total reports:", len(result['data']))

if result['data']:
    first_report = result['data'][0]
    print("\nFirst report:")
    print(f"  Total Cases: {first_report.get('total_cases')}")
    print(f"  Low Severity: {first_report.get('low_severity_count')}")
    print(f"  Medium Severity: {first_report.get('medium_severity_count')}")
    print(f"  High Severity: {first_report.get('high_severity_count')}")
    
    print("\n  Violated_rules:")
    print("    Type:", type(first_report['violated_rules']))
    print("    Is list:", isinstance(first_report['violated_rules'], list))
    
    if isinstance(first_report['violated_rules'], list):
        print("    Count:", len(first_report['violated_rules']))
        if first_report['violated_rules']:
            print("\n    First rule:", first_report['violated_rules'][0])
    else:
        print("    Value:", first_report['violated_rules'][:100])
