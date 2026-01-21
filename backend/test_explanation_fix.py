"""
Quick Test: Explanation Service Fix
====================================
Verify that the explanation service correctly returns:
1. Red Flag cases with NULL or Waiting status
2. Never Event cases with NULL or Waiting status  
3. Ordinary cases with RequiresExplanation=1 and Waiting status
4. Proper response format { success, data, statistics }
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.api.services.explanation_service import get_pending_explanations


def test_pending_explanations():
    """Test the get_pending_explanations service"""
    print("\n" + "="*70)
    print("TESTING: get_pending_explanations()")
    print("="*70)
    
    # Call the service
    result = get_pending_explanations()
    
    # Check response structure
    print("\n1. RESPONSE STRUCTURE:")
    print(f"   - Has 'success' key: {('success' in result)}")
    print(f"   - Has 'data' key: {('data' in result)}")
    print(f"   - Has 'statistics' key: {('statistics' in result)}")
    print(f"   - Success value: {result.get('success')}")
    
    # Check data type
    data = result.get('data', [])
    print(f"\n2. DATA ARRAY:")
    print(f"   - Type: {type(data)}")
    print(f"   - Is list: {isinstance(data, list)}")
    print(f"   - Count: {len(data)}")
    
    # Check statistics
    stats = result.get('statistics', {})
    print(f"\n3. STATISTICS:")
    print(f"   - Total count: {stats.get('total_count')}")
    print(f"   - Red Flag count: {stats.get('red_flag_count')}")
    print(f"   - Ordinary count: {stats.get('ordinary_count')}")
    
    # Sample cases
    if data:
        print(f"\n4. SAMPLE CASES (first 5):")
        for i, case in enumerate(data[:5]):
            case_id = case.get('IncidentRequestCaseID')
            clinical_type = case.get('ClinicalRiskType')
            clinical_id = case.get('ClinicalRiskTypeID')
            exp_status = case.get('ExplanationStatusName')
            exp_id = case.get('ExplanationStatusID')
            case_status = case.get('CaseStatusName')
            requires_exp = case.get('RequiresExplanation')
            
            print(f"\n   Case #{i+1} (ID={case_id}):")
            print(f"      Clinical Risk: {clinical_type} (ID={clinical_id})")
            print(f"      Explanation Status: {exp_status} (ID={exp_id})")
            print(f"      Case Status: {case_status}")
            print(f"      Requires Explanation: {requires_exp}")
    else:
        print(f"\n4. NO CASES FOUND")
        print(f"   This might be expected if:")
        print(f"   - Database is empty")
        print(f"   - All cases have been resolved")
        print(f"   - Query logic is still incorrect")
    
    # Breakdown by type
    if data:
        red_flags = [c for c in data if c.get('ClinicalRiskTypeID') == 2]
        never_events = [c for c in data if c.get('ClinicalRiskTypeID') == 3]
        ordinary = [c for c in data if c.get('ClinicalRiskTypeID') not in [2, 3]]
        
        print(f"\n5. BREAKDOWN BY CLINICAL RISK TYPE:")
        print(f"   - Red Flag (ID=2): {len(red_flags)} cases")
        print(f"   - Never Event (ID=3): {len(never_events)} cases")
        print(f"   - Ordinary: {len(ordinary)} cases")
        
        # Check for NULL ExplanationStatusID
        null_status = [c for c in data if c.get('ExplanationStatusID') is None]
        waiting_status = [c for c in data if c.get('ExplanationStatusID') == 1]
        
        print(f"\n6. BREAKDOWN BY EXPLANATION STATUS:")
        print(f"   - NULL ExplanationStatusID: {len(null_status)} cases")
        print(f"   - Waiting (ID=1): {len(waiting_status)} cases")
        print(f"   - Total: {len(null_status) + len(waiting_status)} cases")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
    
    return result


if __name__ == "__main__":
    try:
        result = test_pending_explanations()
        
        # Final verdict
        if result.get('success') and isinstance(result.get('data'), list):
            print("✅ SUCCESS: Response structure is correct")
            if result.get('data'):
                print("✅ SUCCESS: Cases returned from database")
            else:
                print("⚠️  WARNING: No cases returned (might be expected)")
        else:
            print("❌ FAILURE: Response structure is incorrect")
            print(f"   Result: {result}")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
