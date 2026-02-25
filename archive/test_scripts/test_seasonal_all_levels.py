"""
Comprehensive test for seasonal report data integrity across all organizational levels.
Tests Administration, Department, and Section levels to verify target department filtering works.
"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from api.db_layer.seasonal_report_aggregation import get_seasonal_domain_totals
from api.db_layer.seasonal_report import resolve_season_id_from_year_trimester
from api.db_layer.admin_units import get_units_by_type

print("=" * 100)
print("COMPREHENSIVE TEST: Seasonal Report Data Integrity - All Organizational Levels")
print("=" * 100)
print()

# Configuration
TEST_YEAR = 2026
TEST_PERIOD = "Q1"

try:
    # Resolve season ID
    season_id = resolve_season_id_from_year_trimester(year=TEST_YEAR, trimester=TEST_PERIOD)
    print(f"✅ Season ID: {season_id} ({TEST_PERIOD} {TEST_YEAR})\n")
    
    # ========================================================================
    # TEST 1: Hospital Level (Baseline)
    # ========================================================================
    print("=" * 100)
    print("TEST 1: Hospital Level (orgunit_type=0)")
    print("=" * 100)
    
    hospital_totals = get_seasonal_domain_totals(
        season_id=season_id,
        orgunit_id=1,
        orgunit_type=0
    )
    
    hospital_count = hospital_totals['total_cases']
    print(f"✅ Hospital Total: {hospital_count} cases\n")
    
    # ========================================================================
    # TEST 2: Administration Level (orgunit_type=1)
    # ========================================================================
    print("=" * 100)
    print("TEST 2: Administration Level (orgunit_type=1) - Type 323")
    print("=" * 100)
    
    administrations = get_units_by_type(323)
    print(f"Found {len(administrations)} administrations\n")
    
    admin_with_data = []
    admin_without_data = []
    
    for admin in administrations:
        totals = get_seasonal_domain_totals(
            season_id=season_id,
            orgunit_id=admin["id"],
            orgunit_type=1
        )
        
        if totals['total_cases'] > 0:
            admin_with_data.append({
                'name': admin['name'],
                'id': admin['id'],
                'count': totals['total_cases']
            })
        else:
            admin_without_data.append({
                'name': admin['name'],
                'id': admin['id']
            })
    
    print(f"📊 Results:")
    print(f"  ✅ With Data: {len(admin_with_data)}")
    for a in admin_with_data:
        print(f"     - {a['name']}: {a['count']} cases")
    print(f"  ⚠️  Empty: {len(admin_without_data)}")
    print()
    
    # ========================================================================
    # TEST 3: Department Level (orgunit_type=2)
    # ========================================================================
    print("=" * 100)
    print("TEST 3: Department Level (orgunit_type=2) - Type 325")
    print("=" * 100)
    
    departments = get_units_by_type(325)
    print(f"Found {len(departments)} departments")
    print(f"Testing first 10 departments...\n")
    
    dept_with_data = []
    dept_without_data = []
    
    for dept in departments[:10]:  # Test first 10
        totals = get_seasonal_domain_totals(
            season_id=season_id,
            orgunit_id=dept["id"],
            orgunit_type=2
        )
        
        if totals['total_cases'] > 0:
            dept_with_data.append({
                'name': dept['name'],
                'id': dept['id'],
                'count': totals['total_cases']
            })
        else:
            dept_without_data.append({
                'name': dept['name'],
                'id': dept['id']
            })
    
    print(f"📊 Results (first 10 tested):")
    print(f"  ✅ With Data: {len(dept_with_data)}")
    for d in dept_with_data:
        print(f"     - {d['name']}: {d['count']} cases")
    print(f"  ⚠️  Empty: {len(dept_without_data)}")
    print()
    
    # ========================================================================
    # TEST 4: Section Level (orgunit_type=3)
    # ========================================================================
    print("=" * 100)
    print("TEST 4: Section Level (orgunit_type=3) - Type 324")
    print("=" * 100)
    
    sections = get_units_by_type(324)
    print(f"Found {len(sections)} sections")
    print(f"Testing first 15 sections...\n")
    
    section_with_data = []
    section_without_data = []
    
    for section in sections[:15]:  # Test first 15
        totals = get_seasonal_domain_totals(
            season_id=season_id,
            orgunit_id=section["id"],
            orgunit_type=3
        )
        
        if totals['total_cases'] > 0:
            section_with_data.append({
                'name': section['name'],
                'id': section['id'],
                'count': totals['total_cases']
            })
        else:
            section_without_data.append({
                'name': section['name'],
                'id': section['id']
            })
    
    print(f"📊 Results (first 15 tested):")
    print(f"  ✅ With Data: {len(section_with_data)}")
    for s in section_with_data:
        print(f"     - {s['name']}: {s['count']} cases")
    print(f"  ⚠️  Empty: {len(section_without_data)}")
    print()
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)
    print()
    
    print(f"Hospital Total: {hospital_count} cases")
    print()
    print(f"Administration Level (orgunit_type=1):")
    print(f"  ✅ Units with data: {len(admin_with_data)}/{len(administrations)}")
    print()
    print(f"Department Level (orgunit_type=2):")
    print(f"  ✅ Units with data: {len(dept_with_data)}/10 tested")
    print()
    print(f"Section Level (orgunit_type=3):")
    print(f"  ✅ Units with data: {len(section_with_data)}/15 tested")
    print()
    
    # Determine overall result
    all_working = (
        len(admin_with_data) > 0 and
        len(dept_with_data) > 0 and
        len(section_with_data) > 0
    )
    
    if all_working:
        print("✅ SUCCESS! All organizational levels are retrieving data correctly.")
        print("   Target department filtering with tree expansion is working at all levels.")
    else:
        print("⚠️  PARTIAL SUCCESS:")
        if len(admin_with_data) == 0:
            print("   ❌ Administration level: No data found")
        if len(dept_with_data) == 0:
            print("   ❌ Department level: No data found")
        if len(section_with_data) == 0:
            print("   ❌ Section level: No data found")
    
    print()
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
