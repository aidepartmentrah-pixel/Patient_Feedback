"""
PHASE K — SVC3 — Quick Verification

Demonstration of migration_text_builder in action with realistic data.
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from api.services.migration_text_builder import build_migration_texts


def demonstrate():
    """Show text builder with realistic legacy data"""
    print("=" * 80)
    print("PHASE K — SVC3 — MIGRATION TEXT BUILDER DEMONSTRATION")
    print("=" * 80)
    
    # Simulate legacy data
    case_row = {
        "Description": "Patient reported severe pain in lower back after surgery. Pain started 2 days post-op and has been increasing in intensity. Patient rates pain as 8/10."
    }
    
    request_row = {
        "Note": "Family member contacted nursing station regarding patient discomfort and requested immediate medical review. Concerned about post-operative complications."
    }
    
    actions = [
        {
            "Description": "Nursing assessment completed",
            "SectionNote": "Vital signs checked - all within normal range",
            "SelectionNote": "Pain medication schedule reviewed",
            "ProblemReason": "Post-operative pain management issue",
            "DateAndTimeCreated": "2025-11-27 10:15:00"
        },
        {
            "Description": "Doctor consultation performed",
            "Note": "Attending physician examined patient",
            "DepartmentNote": "Orthopedic department notified",
            "SectionNote": "X-rays ordered to rule out complications",
            "DateAndTimeCreated": "2025-11-27 11:30:00"
        },
        {
            "Description": "Diagnostic imaging completed",
            "Note": "X-ray results reviewed by radiologist",
            "DepartmentNote": "No fractures or dislocations visible",
            "DateAndTimeCreated": "2025-11-27 14:00:00"
        },
        {
            "Description": "Treatment plan updated",
            "Note": "Pain medication adjusted",
            "GoverningPolicies": "Hospital pain management protocol v2.3",
            "SectionNote": "Patient education on post-op recovery provided",
            "DateAndTimeCreated": "2025-11-27 16:45:00"
        }
    ]
    
    print("\n📥 INPUT DATA:")
    print("-" * 80)
    print(f"Case Description: {case_row['Description'][:60]}...")
    print(f"Request Note: {request_row['Note'][:60]}...")
    print(f"Actions: {len(actions)} records")
    
    # Build migration texts
    result = build_migration_texts(case_row, request_row, actions)
    
    # Display results
    print("\n" + "=" * 80)
    print("📤 OUTPUT — COMPLAINT_CONTENT")
    print("=" * 80)
    print(result["complaint_content"])
    
    print("\n" + "=" * 80)
    print("📤 OUTPUT — IMMEDIATE_ACTION")
    print("=" * 80)
    print(result["immediate_action"])
    
    print("\n" + "=" * 80)
    print("📤 OUTPUT — ACTIONS_TAKEN")
    print("=" * 80)
    print(result["actions_taken"])
    
    # Stats
    print("\n" + "=" * 80)
    print("📊 STATISTICS")
    print("=" * 80)
    print(f"Complaint Content: {len(result['complaint_content'])} characters")
    print(f"Immediate Action:  {len(result['immediate_action'])} characters")
    print(f"Actions Taken:     {len(result['actions_taken'])} characters")
    print(f"Total Output:      {sum(len(v) for v in result.values())} characters")
    
    print("\n" + "=" * 80)
    print("✅ K-SVC-3 — migration_text_builder — FUNCTIONAL")
    print("=" * 80)


if __name__ == "__main__":
    demonstrate()
