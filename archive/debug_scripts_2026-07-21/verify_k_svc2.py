"""
PHASE K — SVC2 — Quick Verification

Quick demo of get_legacy_case_detail functionality
"""

import sys
import json
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.db_layer.legacy_migration_db import get_legacy_case_detail
from core.database import get_connection


def demonstrate():
    """Show function in action"""
    print("=" * 80)
    print("PHASE K — SVC2 — LEGACY CASE DETAIL DEMONSTRATION")
    print("=" * 80)
    
    # Get a test case ID
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT TOP 1 UniqueID FROM IncidentRequestCase ORDER BY UniqueID")
    case_id = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    
    print(f"\n📋 Retrieving detail for legacy case ID: {case_id}")
    
    result = get_legacy_case_detail(case_id)
    
    if not result:
        print("❌ Case not found")
        return
    
    # Display case info
    print("\n" + "=" * 80)
    print("CASE DATA")
    print("=" * 80)
    case = result["case"]
    print(f"  Case ID: {case['UniqueID']}")
    print(f"  Description: {case['Description'][:100] if case['Description'] else 'N/A'}...")
    print(f"  Note: {case['Note'][:50] if case['Note'] else 'N/A'}")
    print(f"  Created: {case['DateAndTimeCreated']}")
    print(f"  Updated: {case['DateAndTimeUpdated']}")
    print(f"  Happened: {case['DateAndTimeHappened']}")
    print(f"  Doctor ID: {case['DoctorID']}")
    print(f"  Section ID: {case['SectionID']}")
    print(f"  Department ID: {case['DepartmentID']}")
    print(f"  Admin ID: {case['AdminID']}")
    
    # Display request info
    print("\n" + "=" * 80)
    print("REQUEST DATA")
    print("=" * 80)
    request = result["request"]
    print(f"  Patient Name: {request['PatientName']}")
    print(f"  MRN: {request['MRN']}")
    print(f"  Source Building: {request['SourceBuilding']}")
    print(f"  Is Inpatient: {request['IsInPatient']}")
    print(f"  Requester: {request['RequesterName']}")
    print(f"  Note: {request['Note'][:50] if request['Note'] else 'N/A'}")
    print(f"  Received: {request['DateAndTimeRecieved']}")
    print(f"  Source Section: {request['SourceSectionID']}")
    print(f"  Source Department: {request['SourceDepartmentID']}")
    print(f"  Source Admin: {request['SourceAdminID']}")
    
    # Display actions
    print("\n" + "=" * 80)
    print("ACTION HISTORY")
    print("=" * 80)
    actions = result["actions"]
    
    if not actions:
        print("  No actions recorded")
    else:
        print(f"  Total actions: {len(actions)}\n")
        for i, action in enumerate(actions, 1):
            print(f"  {i}. Action ID: {action['UniqueID']}")
            print(f"     Date: {action['DateAndTimeCreated']}")
            print(f"     Description: {action['Description']}")
            if action['Note']:
                print(f"     Note: {action['Note'][:60]}...")
            print()
    
    print("=" * 80)
    print("✅ K-SVC-2 — get_legacy_case_detail — FUNCTIONAL")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate()
