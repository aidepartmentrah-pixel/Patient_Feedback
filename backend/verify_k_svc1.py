"""
PHASE K — SVC1 — Quick Verification

Quick demo of list_legacy_cases_paged functionality
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.db_layer.legacy_migration_db import list_legacy_cases_paged


def demonstrate():
    """Show function in action"""
    print("=" * 80)
    print("PHASE K — SVC1 — LEGACY PAGED LIST DEMONSTRATION")
    print("=" * 80)
    
    # Get first page
    print("\n📄 Page 1 (5 per page):")
    rows, total = list_legacy_cases_paged(page=1, page_size=5)
    
    print(f"   Total unmigrated cases: {total}")
    print(f"   Returned: {len(rows)} rows\n")
    
    for i, row in enumerate(rows, 1):
        print(f"   {i}. Case ID: {row['legacy_case_id']}")
        print(f"      Patient: {row['patient_name']}")
        print(f"      Date: {row['received_date']}")
        print(f"      Preview: {row['preview_description'][:60]}...")
        print()
    
    # Get second page
    if total > 5:
        print("\n📄 Page 2 (5 per page):")
        rows2, total2 = list_legacy_cases_paged(page=2, page_size=5)
        
        print(f"   Returned: {len(rows2)} rows\n")
        
        for i, row in enumerate(rows2, 1):
            print(f"   {i}. Case ID: {row['legacy_case_id']}")
            print(f"      Patient: {row['patient_name']}")
            print()
    
    print("=" * 80)
    print("✅ K-SVC-1 — list_legacy_cases_paged — FUNCTIONAL")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate()
