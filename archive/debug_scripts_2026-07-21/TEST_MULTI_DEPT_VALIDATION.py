"""
Multi-Department Count Validation Test
Validates count integrity across all organizational levels with proper multi-target handling.
"""

import sys
from datetime import date
from collections import defaultdict

sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.services.monthly_report_service import monthly_report_service
from api.db_layer.admin_units import get_units_by_type


def comprehensive_count_validation(year: int, month: int):
    """
    Comprehensive validation across all organizational levels.
    """
    print("=" * 100)
    print(f"COMPREHENSIVE COUNT VALIDATION - {year}-{month:02d}")
    print("=" * 100)
    print()
    
    # ========== STEP 1: Hospital Level (Baseline) ==========
    print("📊 STEP 1: Hospital-Wide Baseline")
    print("-" * 100)
    
    hospital_result = monthly_report_service.generate_monthly_report(
        year=year, month=month, start_date=None, end_date=None,
        mode="detailed", scope=None,
        administration_ids=None, department_ids=None, section_ids=None,
        page=1, page_size=9999
    )
    
    hospital_complaints = hospital_result.get("complaints", [])
    hospital_total = len(hospital_complaints)
    hospital_ids = set(c["id"] for c in hospital_complaints)
    
    # Analyze target departments
    total_target_pairs = 0
    complaints_by_target_count = defaultdict(int)
    
    for c in hospital_complaints:
        targets = c.get("target_departments", [])
        target_count = len(targets)
        total_target_pairs += target_count
        complaints_by_target_count[target_count] += 1
    
    avg_targets = total_target_pairs / hospital_total if hospital_total > 0 else 0
    
    print(f"✓ Unique Complaints (Hospital): {hospital_total}")
    print(f"✓ Total Complaint-Department Pairs: {total_target_pairs}")
    print(f"✓ Average Targets per Complaint: {avg_targets:.2f}")
    print()
    print("Distribution by Target Count:")
    for target_count in sorted(complaints_by_target_count.keys()):
        count = complaints_by_target_count[target_count]
        print(f"  - {target_count} target(s): {count} complaints")
    print()
    
    # ========== STEP 2: Administration Level ==========
    print("📊 STEP 2: Administration-Level Validation")
    print("-" * 100)
    
    administrations = get_units_by_type(323)  # Type 323 = Administration
    print(f"Testing {len(administrations)} administrations...")
    print()
    
    admin_results = {}
    all_admin_complaint_ids = set()
    sum_admin_counts = 0
    
    for admin in administrations:
        admin_id = admin["id"]
        admin_name = admin["name"]
        
        result = monthly_report_service.generate_monthly_report(
            year=year, month=month, start_date=None, end_date=None,
            mode="detailed", scope=None,
            administration_ids=str(admin_id), department_ids=None, section_ids=None,
            page=1, page_size=9999
        )
        
        complaints = result.get("complaints", [])
        complaint_ids = [c["id"] for c in complaints]
        unique_ids = set(complaint_ids)
        duplicates = len(complaint_ids) - len(unique_ids)
        
        admin_results[admin_id] = {
            "name": admin_name,
            "total": len(complaints),
            "unique": len(unique_ids),
            "duplicates": duplicates,
            "ids": unique_ids
        }
        
        all_admin_complaint_ids.update(unique_ids)
        sum_admin_counts += len(complaints)
        
        status = "✓" if duplicates == 0 else "❌"
        if len(complaints) > 0:
            print(f"{status} {admin_name[:50]:50s} Count: {len(complaints):3d} | Unique: {len(unique_ids):3d} | Dupes: {duplicates}")
    
    print()
    print(f"Sum of All Admin Reports: {sum_admin_counts}")
    print(f"Unique Complaints Across All Admins: {len(all_admin_complaint_ids)}")
    print()
    
    # ========== STEP 3: Department Level ==========
    print("📊 STEP 3: Department-Level Validation")
    print("-" * 100)
    
    departments = get_units_by_type(325)  # Type 325 = Department
    print(f"Testing {len(departments)} departments...")
    print()
    
    dept_results = {}
    all_dept_complaint_ids = set()
    sum_dept_counts = 0
    
    for dept in departments:
        dept_id = dept["id"]
        dept_name = dept["name"]
        
        result = monthly_report_service.generate_monthly_report(
            year=year, month=month, start_date=None, end_date=None,
            mode="detailed", scope=None,
            administration_ids=None, department_ids=str(dept_id), section_ids=None,
            page=1, page_size=9999
        )
        
        complaints = result.get("complaints", [])
        complaint_ids = [c["id"] for c in complaints]
        unique_ids = set(complaint_ids)
        duplicates = len(complaint_ids) - len(unique_ids)
        
        dept_results[dept_id] = {
            "name": dept_name,
            "total": len(complaints),
            "unique": len(unique_ids),
            "duplicates": duplicates,
            "ids": unique_ids
        }
        
        all_dept_complaint_ids.update(unique_ids)
        sum_dept_counts += len(complaints)
        
        if len(complaints) > 0:
            status = "✓" if duplicates == 0 else "❌"
            print(f"{status} {dept_name[:50]:50s} Count: {len(complaints):3d} | Unique: {len(unique_ids):3d} | Dupes: {duplicates}")
    
    print()
    print(f"Sum of All Dept Reports: {sum_dept_counts}")
    print(f"Unique Complaints Across All Depts: {len(all_dept_complaint_ids)}")
    print()
    
    # ========== STEP 4: Section Level ==========
    print("📊 STEP 4: Section-Level Validation")
    print("-" * 100)
    
    sections = get_units_by_type(324)  # Type 324 = Section
    print(f"Testing {len(sections)} sections...")
    print()
    
    section_results = {}
    all_section_complaint_ids = set()
    sum_section_counts = 0
    
    for section in sections:
        section_id = section["id"]
        section_name = section["name"]
        
        result = monthly_report_service.generate_monthly_report(
            year=year, month=month, start_date=None, end_date=None,
            mode="detailed", scope=None,
            administration_ids=None, department_ids=None, section_ids=str(section_id),
            page=1, page_size=9999
        )
        
        complaints = result.get("complaints", [])
        complaint_ids = [c["id"] for c in complaints]
        unique_ids = set(complaint_ids)
        duplicates = len(complaint_ids) - len(unique_ids)
        
        section_results[section_id] = {
            "name": section_name,
            "total": len(complaints),
            "unique": len(unique_ids),
            "duplicates": duplicates,
            "ids": unique_ids
        }
        
        all_section_complaint_ids.update(unique_ids)
        sum_section_counts += len(complaints)
        
        if len(complaints) > 0:
            status = "✓" if duplicates == 0 else "❌"
            print(f"{status} {section_name[:50]:50s} Count: {len(complaints):3d} | Unique: {len(unique_ids):3d} | Dupes: {duplicates}")
    
    print()
    print(f"Sum of All Section Reports: {sum_section_counts}")
    print(f"Unique Complaints Across All Sections: {len(all_section_complaint_ids)}")
    print()
    
    # ========== FINAL VALIDATION ==========
    print("=" * 100)
    print("🔍 FINAL VALIDATION RESULTS")
    print("=" * 100)
    print()
    
    issues = []
    
    # Summary Table
    print("Level                      | Total Records | Unique Complaints | Duplicates")
    print("-" * 100)
    print(f"Hospital (Baseline)        | {hospital_total:13d} | {hospital_total:17d} | {'0':10s}")
    print(f"All Administrations (Sum)  | {sum_admin_counts:13d} | {len(all_admin_complaint_ids):17d} | {'N/A':10s}")
    print(f"All Departments (Sum)      | {sum_dept_counts:13d} | {len(all_dept_complaint_ids):17d} | {'N/A':10s}")
    print(f"All Sections (Sum)         | {sum_section_counts:13d} | {len(all_section_complaint_ids):17d} | {'N/A':10s}")
    print()
    
    # Validation Checks
    print("Validation Checks:")
    print("-" * 100)
    
    # Check 1: No duplicates within individual reports
    all_results = list(admin_results.values()) + list(dept_results.values()) + list(section_results.values())
    duplicate_reports = [r for r in all_results if r["duplicates"] > 0]
    
    if duplicate_reports:
        issues.append(f"Found {len(duplicate_reports)} reports with duplicate complaints")
        print(f"❌ FAIL: {len(duplicate_reports)} reports have duplicate complaints within them")
    else:
        print("✓ PASS: No duplicate complaints within individual reports")
    
    # Check 2: Expected relationship with multi-targeting
    expected_min = hospital_total
    expected_max = total_target_pairs
    
    print(f"\n✓ Multi-Target Accounting:")
    print(f"  - Hospital unique complaints: {hospital_total}")
    print(f"  - Total complaint-target pairs: {total_target_pairs}")
    print(f"  - Expected sum range: {expected_min} to {expected_max}")
    
    # Check Administration sum
    if expected_min <= sum_admin_counts <= expected_max:
        print(f"  ✓ PASS: Admin sum ({sum_admin_counts}) is within expected range")
    else:
        issues.append(f"Admin sum ({sum_admin_counts}) outside expected range [{expected_min}, {expected_max}]")
        print(f"  ❌ FAIL: Admin sum ({sum_admin_counts}) outside expected range")
    
    # Check Department sum
    if expected_min <= sum_dept_counts <= expected_max:
        print(f"  ✓ PASS: Dept sum ({sum_dept_counts}) is within expected range")
    else:
        issues.append(f"Dept sum ({sum_dept_counts}) outside expected range [{expected_min}, {expected_max}]")
        print(f"  ❌ FAIL: Dept sum ({sum_dept_counts}) outside expected range")
    
    # Check Section sum (should equal total_target_pairs since sections are leaf nodes)
    if sum_section_counts == expected_max:
        print(f"  ✓ PASS: Section sum ({sum_section_counts}) equals total target pairs ({expected_max})")
    elif sum_section_counts < expected_max:
        diff = expected_max - sum_section_counts
        print(f"  ⚠️  WARNING: Section sum ({sum_section_counts}) is {diff} less than expected ({expected_max})")
        print(f"      This might indicate orphaned target departments or missing sections")
    else:
        issues.append(f"Section sum ({sum_section_counts}) > total target pairs ({expected_max})")
        print(f"  ❌ FAIL: Section sum ({sum_section_counts}) exceeds total target pairs ({expected_max})")
    
    # Check 3: Coverage (all hospital complaints appear somewhere)
    if len(all_section_complaint_ids) == hospital_total:
        print(f"\n✓ PASS: All {hospital_total} hospital complaints appear in section reports")
    else:
        missing = hospital_total - len(all_section_complaint_ids)
        issues.append(f"{missing} complaints missing from section reports")
        print(f"\n❌ FAIL: {missing} complaints missing from section reports")
        
        # Find which complaints are missing
        missing_ids = hospital_ids - all_section_complaint_ids
        print(f"  Missing complaint IDs: {sorted(list(missing_ids))[:10]}...")
    
    # Final Summary
    print()
    print("=" * 100)
    if issues:
        print("❌ DATA INTEGRITY ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ ALL INTEGRITY CHECKS PASSED!")
        print("✅ Multi-department targeting is correctly handled")
        print("✅ Count validation confirmed across all levels")
    print("=" * 100)


if __name__ == "__main__":
    # Test with December 2025
    comprehensive_count_validation(year=2025, month=12)
