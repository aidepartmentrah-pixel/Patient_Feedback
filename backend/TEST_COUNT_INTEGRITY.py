"""
Count Integrity Validation Test
Tests that complaint counts are accurate and consistent across different report levels.
"""

import sys
from datetime import date
from collections import defaultdict

# Add backend to path
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.services.monthly_report_service import monthly_report_service
from api.db_layer.admin_units import get_units_by_type
from api.db_layer.reports_db import get_filtered_complaints


def test_count_integrity(year: int, month: int):
    """
    Test complaint count integrity across different organizational levels.
    """
    print("=" * 80)
    print(f"COUNT INTEGRITY TEST - {year}-{month:02d}")
    print("=" * 80)
    print()
    
    # ========== STEP 1: Get Hospital Total ==========
    print("📊 STEP 1: Hospital-Wide Count")
    print("-" * 80)
    
    hospital_result = monthly_report_service.generate_monthly_report(
        year=year,
        month=month,
        start_date=None,
        end_date=None,
        mode="detailed",
        scope=None,
        administration_ids=None,
        department_ids=None,
        section_ids=None,
        page=1,
        page_size=9999
    )
    
    hospital_complaints = hospital_result.get("complaints", [])
    hospital_total = len(hospital_complaints)
    hospital_ids = set(c["id"] for c in hospital_complaints)
    
    print(f"✓ Total Complaints (Hospital-Wide): {hospital_total}")
    print(f"✓ Unique Complaint IDs: {len(hospital_ids)}")
    print()
    
    # ========== STEP 2: Test Administration Level ==========
    print("📊 STEP 2: Administration-Level Counts")
    print("-" * 80)
    
    # Type 323 = Administration
    administrations = get_units_by_type(323)
    print(f"Found {len(administrations)} administrations")
    print()
    
    admin_complaint_map = defaultdict(set)  # admin_id -> set of complaint IDs
    admin_counts = {}
    
    for admin in administrations:  # Test ALL administrations
        admin_id = admin["id"]
        admin_name = admin["name"]
        
        result = monthly_report_service.generate_monthly_report(
            year=year,
            month=month,
            start_date=None,
            end_date=None,
            mode="detailed",
            scope=None,
            administration_ids=str(admin_id),
            department_ids=None,
            section_ids=None,
            page=1,
            page_size=9999
        )
        
        complaints = result.get("complaints", [])
        complaint_ids = [c["id"] for c in complaints]
        unique_ids = set(complaint_ids)
        
        # Check for duplicates within this report
        duplicates = len(complaint_ids) - len(unique_ids)
        
        admin_complaint_map[admin_id] = unique_ids
        admin_counts[admin_id] = {
            "name": admin_name,
            "total": len(complaints),
            "unique": len(unique_ids),
            "duplicates": duplicates
        }
        
        status = "✓" if duplicates == 0 else "❌"
        print(f"{status} {admin_name[:40]:40s} - Count: {len(complaints):3d} | Unique: {len(unique_ids):3d} | Dupes: {duplicates}")
    
    print()
    
    # ========== STEP 3: Check for Multi-Targeting ==========
    print("📊 STEP 3: Multi-Target Analysis")
    print("-" * 80)
    
    multi_target_complaints = []
    for complaint in hospital_complaints[:50]:  # Sample first 50
        target_depts = complaint.get("target_departments", [])
        if len(target_depts) > 1:
            multi_target_complaints.append({
                "id": complaint["id"],
                "target_count": len(target_depts),
                "targets": [d.get("section_name") or d.get("department_name") for d in target_depts]
            })
    
    print(f"✓ Complaints with Multiple Targets (sample of 50): {len(multi_target_complaints)}")
    if multi_target_complaints:
        print("\nExamples:")
        for i, mc in enumerate(multi_target_complaints[:3], 1):
            print(f"  {i}. Complaint #{mc['id']} targets {mc['target_count']} departments:")
            for target in mc['targets'][:3]:
                print(f"     - {target}")
    print()
    
    # ========== STEP 4: Count Sum Validation ==========
    print("📊 STEP 4: Count Sum Validation")
    print("-" * 80)
    
    # Union of all complaints across tested administrations
    all_admin_complaints = set()
    for complaint_set in admin_complaint_map.values():
        all_admin_complaints.update(complaint_set)
    
    sum_admin_counts = sum(data["total"] for data in admin_counts.values())
    unique_across_admins = len(all_admin_complaints)
    
    print(f"Hospital Total:                    {hospital_total}")
    print(f"Sum of Admin Reports (tested):     {sum_admin_counts}")
    print(f"Unique Complaints Across Admins:   {unique_across_admins}")
    print()
    
    # Validation
    print("🔍 VALIDATION RESULTS:")
    print("-" * 80)
    
    issues = []
    
    # Check 1: No duplicates within individual reports
    duplicate_reports = [data for data in admin_counts.values() if data["duplicates"] > 0]
    if duplicate_reports:
        issues.append(f"❌ FAIL: {len(duplicate_reports)} reports have duplicate complaints")
        for report in duplicate_reports:
            print(f"   - {report['name']}: {report['duplicates']} duplicates")
    else:
        print("✓ PASS: No duplicate complaints within individual reports")
    
    # Check 2: Sum can be >= hospital total (due to multi-targeting)
    if sum_admin_counts > hospital_total:
        print(f"✓ EXPECTED: Sum of admin reports ({sum_admin_counts}) > Hospital total ({hospital_total})")
        print(f"  This is normal due to multi-department targeting")
    elif sum_admin_counts == hospital_total:
        print(f"✓ PASS: Sum equals hospital total (no multi-targeting in tested admins)")
    else:
        issues.append(f"❌ FAIL: Sum of admin reports ({sum_admin_counts}) < Hospital total ({hospital_total})")
        print(f"❌ FAIL: Missing complaints in admin reports!")
    
    # Check 3: Unique complaints across admins should be <= hospital total
    if unique_across_admins <= hospital_total:
        print(f"✓ PASS: Unique complaints across admins ({unique_across_admins}) ≤ Hospital total ({hospital_total})")
    else:
        issues.append(f"❌ FAIL: More unique complaints in admins than hospital total")
        print(f"❌ FAIL: Unique across admins ({unique_across_admins}) > Hospital total ({hospital_total})")
    
    print()
    print("=" * 80)
    if issues:
        print("❌ DATA INTEGRITY ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ ALL INTEGRITY CHECKS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    # Test for December 2025 (has 41 complaints)
    test_count_integrity(year=2025, month=12)
