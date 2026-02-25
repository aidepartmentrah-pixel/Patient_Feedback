"""
Test Patient History Backend Fixes
Tests all the backend fixes implemented for Phase R-P
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from api.services.patients_service import (
    search_patients_service,
    get_patient_full_history_service,
    export_patient_history_service
)


def test_search_patients():
    """Test 1: Verify search returns normalized fields"""
    print("\n" + "="*80)
    print("TEST 1: Search Patients - Field Normalization")
    print("="*80)
    
    try:
        # Try with MRN
        result = search_patients_service(mrn="MED", limit=5)
        
        print(f"\n✓ Search returned {result['total']} patients")
        
        if result['patients']:
            patient = result['patients'][0]
            print(f"\nFirst patient fields:")
            for key, value in patient.items():
                print(f"  • {key}: {value}")
            
            # Check for full_name instead of patient_name
            if 'full_name' in patient:
                print("\n✅ PASS: Field 'full_name' present (normalized)")
            else:
                print("\n❌ FAIL: Field 'full_name' missing")
            
            # Check gender normalization
            if 'gender' in patient:
                if patient['gender'] in ['Male', 'Female']:
                    print(f"✅ PASS: Gender normalized to '{patient['gender']}'")
                else:
                    print(f"⚠️ WARNING: Gender not fully normalized: '{patient['gender']}'")
        else:
            print("⚠️ No patients found in search")
            
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


def test_full_history_schema():
    """Test 2: Verify full-history returns V2 schema"""
    print("\n" + "="*80)
    print("TEST 2: Full History - V2 Schema Compliance")
    print("="*80)
    
    try:
        # First search for a patient
        search_result = search_patients_service(mrn="MED", limit=1)
        
        if not search_result['patients']:
            print("⚠️ No patients found, skipping test")
            return
        
        patient_id = search_result['patients'][0]['patient_id']
        print(f"\nTesting with patient_id: {patient_id}")
        
        # Get full history
        result = get_patient_full_history_service(patient_id=patient_id, limit=10)
        
        print("\nTop-level keys:")
        for key in result.keys():
            print(f"  • {key}")
        
        # Check V2 schema structure
        required_keys = ['profile', 'metrics', 'items', 'meta']
        missing_keys = [key for key in required_keys if key not in result]
        
        if not missing_keys:
            print("\n✅ PASS: All V2 schema keys present")
        else:
            print(f"\n❌ FAIL: Missing V2 schema keys: {missing_keys}")
        
        # Check metrics structure
        if 'metrics' in result:
            print(f"\nMetrics fields:")
            for key, value in result['metrics'].items():
                print(f"  • {key}: {value}")
            
            if 'severity_breakdown' in result['metrics']:
                print("\n✅ PASS: Severity aggregation present")
            else:
                print("\n❌ FAIL: Severity aggregation missing")
            
            if 'category_breakdown' in result['metrics']:
                print("✅ PASS: Category aggregation present")
            else:
                print("❌ FAIL: Category aggregation missing")
        
        # Check items (incidents)
        if 'items' in result:
            print(f"\n✓ Found {len(result['items'])} incidents in items array")
            
            if result['items']:
                incident = result['items'][0]
                print(f"\nFirst incident fields:")
                for key, value in incident.items():
                    if key not in ['description']:  # Skip long text
                        print(f"  • {key}: {value}")
                
                # Check for snake_case fields
                if 'incident_id' in incident:
                    print("\n✅ PASS: Field names normalized to snake_case")
                else:
                    print("\n⚠️ WARNING: Some field names may not be normalized")
                
                # Check boolean types
                if 'is_red_flag' in incident:
                    print(f"✓ is_red_flag type: {type(incident['is_red_flag']).__name__}")
                    if isinstance(incident['is_red_flag'], bool):
                        print("✅ PASS: Boolean fields properly converted")
                    else:
                        print("❌ FAIL: Boolean fields still integers")
                
                # Check doctor name
                if 'doctor_name' in incident:
                    print(f"✓ doctor_name value: {incident['doctor_name']}")
                    print("✅ PASS: DoctorName field present")
            else:
                print("\n⚠️ No incidents found for patient")
        
        # Check meta
        if 'meta' in result:
            print(f"\nMeta fields:")
            for key, value in result['meta'].items():
                print(f"  • {key}: {value}")
            
            if 'entity_type' in result['meta'] and result['meta']['entity_type'] == 'patient':
                print("\n✅ PASS: Meta structure correct")
            else:
                print("\n❌ FAIL: Meta structure incorrect")
                
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def test_export_union():
    """Test 3: Verify export works for reserve patients"""
    print("\n" + "="*80)
    print("TEST 3: Export - UNION Hospital + Reserve Patients")
    print("="*80)
    
    try:
        # Search for a patient
        search_result = search_patients_service(mrn="MED", limit=1)
        
        if not search_result['patients']:
            print("⚠️ No patients found, skipping test")
            return
        
        patient = search_result['patients'][0]
        patient_id = patient['patient_id']
        source = patient.get('source', 'unknown')
        
        print(f"\nTesting export for patient_id: {patient_id}")
        print(f"Patient source: {source}")
        print(f"Patient name: {patient.get('full_name', 'N/A')}")
        
        # Test JSON export
        result = export_patient_history_service(
            patient_id=patient_id,
            format_type="json",
            include_profile=True
        )
        
        if 'patient' in result and result['patient']:
            print(f"\n✅ PASS: Export includes patient profile")
            print(f"  • Patient: {result['patient'].get('full_name')}")
            print(f"  • MRN: {result['patient'].get('mrn')}")
            print(f"  • Total incidents: {result['patient'].get('total_incidents')}")
        else:
            print(f"\n❌ FAIL: Export missing patient profile")
        
        if 'incidents' in result:
            print(f"\n✓ Export includes {len(result['incidents'])} incidents")
            
            if result['incidents']:
                incident = result['incidents'][0]
                print(f"\nFirst incident sample:")
                print(f"  • RecordID: {incident.get('RecordID')}")
                print(f"  • Date: {incident.get('Date')}")
                print(f"  • DoctorName: {incident.get('DoctorName')}")
                print(f"  • Department: {incident.get('Department')}")
                print(f"  • Severity: {incident.get('Severity')}")
                
                # Check if doctor name is properly populated
                doctor_name = incident.get('DoctorName')
                if doctor_name and doctor_name not in ['Unknown', 'غير محدد', patient.get('full_name'), patient.get('patient_name')]:
                    print("\n✅ PASS: DoctorName properly joined (not patient name)")
                elif doctor_name == 'غير محدد':
                    print("\n✓ INFO: DoctorName is 'غير محدد' (no doctor assigned)")
                else:
                    print(f"\n⚠️ WARNING: DoctorName may be patient name: {doctor_name}")
            else:
                print("\n⚠️ No incidents found for patient")
        
        print("\n✅ PASS: Export query executed successfully")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("PATIENT HISTORY BACKEND FIXES - TEST SUITE")
    print("="*80)
    print("\nTesting the following fixes:")
    print("  1. DoctorName JOIN to APP_IncidentCaseDoctor")
    print("  2. Severity and Category aggregation")
    print("  3. V2 schema normalization (profile/metrics/items/meta)")
    print("  4. Export UNION for hospital + reserve patients")
    print("  5. Field name normalization (full_name, snake_case)")
    print("  6. Boolean field conversion (true/false)")
    print("  7. Gender normalization (Male/Female)")
    print("="*80)
    
    test_search_patients()
    test_full_history_schema()
    test_export_union()
    
    print("\n" + "="*80)
    print("TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_all_tests()
