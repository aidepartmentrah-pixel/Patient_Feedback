"""
TEST: All 4 ML Training Fields Implementation
==============================================
Tests:
1. API accepts all 4 fields
2. Fields pass through to ML insert
3. Fields are populated in database
"""

import sqlite3
from pathlib import Path
import sys

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.ml_mapping import add_corrected_record_to_ml

ML_DB_PATH = workspace_root / "models_directory" / "patient_feedback_ml.db"

def test_all_four_fields():
    """Test that all 4 new fields are accepted and stored."""
    
    print("\n" + "="*80)
    print("TEST: ALL 4 ML TRAINING FIELDS")
    print("="*80)
    
    # Create test data with all 4 new fields
    test_data = {
        # Text fields (for embeddings)
        "complaint_text": "المريض يشكو من ألم شديد وتأخر في التشخيص",
        "immediate_action": "تم إعطاء مسكنات قوية والتحقق من الحالة",
        "taken_action": "تم نقل المريض إلى قسم الطوارئ",
        
        # Classification
        "domain_id": 1,
        "category_id": 5,
        "subcategory_id": 5,
        "severity_id": 1,
        "stage_id": 1,
        "harm_id": 5,
        
        # Metadata
        "feedback_received_date": "2026-01-02 15:00:00",
        
        # THE 4 NEW FIELDS
        "feedback_type": 1,                          # Option 1: Improvement Opportunity
        "improvement_opportunity_type": 2,           # Option 2: Red Flag
        "classification_ar": 8.5,                    # Option 3: Arabic score
        "classification_en": 5,                      # Option 4: English code
    }
    
    print("\n[STEP 1] Test Data Prepared")
    print(f"  Input fields: {len(test_data)}")
    print(f"  - feedback_type: {test_data['feedback_type']}")
    print(f"  - improvement_opportunity_type: {test_data['improvement_opportunity_type']}")
    print(f"  - classification_ar: {test_data['classification_ar']}")
    print(f"  - classification_en: {test_data['classification_en']}")
    
    # Get current count
    print("\n[STEP 2] Getting Record Count Before Insert")
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patient_feedback_encoded")
        count_before = cursor.fetchone()[0]
        print(f"  Records before: {count_before}")
        conn.close()
    except Exception as e:
        print(f"  [ERROR] Cannot count records: {e}")
        return False
    
    # Insert data
    print("\n[STEP 3] Inserting Data")
    try:
        add_corrected_record_to_ml(test_data)
        print(f"  [OK] Insert function executed")
    except Exception as e:
        print(f"  [ERROR] Insert failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Verify insert
    print("\n[STEP 4] Verifying Insert")
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patient_feedback_encoded")
        count_after = cursor.fetchone()[0]
        print(f"  Records after: {count_after}")
        
        if count_after > count_before:
            print(f"  [OK] {count_after - count_before} new record(s) inserted")
        else:
            print(f"  [ERROR] No records inserted")
            conn.close()
            return False
        
        # Get the new record
        cursor.execute("""
            SELECT * FROM patient_feedback_encoded 
            ORDER BY id DESC LIMIT 1
        """)
        record = cursor.fetchone()
        
        if not record:
            print("  [ERROR] Cannot retrieve inserted record")
            conn.close()
            return False
        
        # Get column names
        cursor.execute("PRAGMA table_info(patient_feedback_encoded)")
        columns = [row[1] for row in cursor.fetchall()]
        record_dict = dict(zip(columns, record))
        
        # Check the 4 new fields
        print("\n[STEP 5] Verifying The 4 New Fields")
        
        all_ok = True
        
        feedback_type = record_dict.get("feedback_type")
        if feedback_type == test_data["feedback_type"]:
            print(f"  [OK] feedback_type: {feedback_type} (1=improvement Opportunity)")
        else:
            print(f"  [ERROR] feedback_type: {feedback_type} (expected {test_data['feedback_type']})")
            all_ok = False
        
        improvement_opp = record_dict.get("improvement_opportunity_type")
        if improvement_opp == test_data["improvement_opportunity_type"]:
            print(f"  [OK] improvement_opportunity_type: {improvement_opp} (2=Red Flag)")
        else:
            print(f"  [ERROR] improvement_opportunity_type: {improvement_opp} (expected {test_data['improvement_opportunity_type']})")
            all_ok = False
        
        classification_ar = record_dict.get("classification_ar")
        if classification_ar == test_data["classification_ar"]:
            print(f"  [OK] classification_ar: {classification_ar}")
        else:
            print(f"  [ERROR] classification_ar: {classification_ar} (expected {test_data['classification_ar']})")
            all_ok = False
        
        classification_en = record_dict.get("classification_en")
        if classification_en == test_data["classification_en"]:
            print(f"  [OK] classification_en: {classification_en}")
        else:
            print(f"  [ERROR] classification_en: {classification_en} (expected {test_data['classification_en']})")
            all_ok = False
        
        # Check embeddings were generated
        print("\n[STEP 6] Verifying Embeddings Generated")
        
        embedding_fields = [
            "embedding_text1", "embedding_text2", "embedding_text3",
            "embedding_text123", "embedding_text23",
            "sentence_1_embedding"
        ]
        
        embeddings_ok = True
        for field in embedding_fields:
            value = record_dict.get(field)
            if value and len(str(value)) > 100:  # Embeddings are long
                print(f"  [OK] {field}: generated (bytes)")
            else:
                print(f"  [ERROR] {field}: NOT generated")
                embeddings_ok = False
        
        conn.close()
        
        # Summary
        print("\n" + "="*80)
        print("RESULT")
        print("="*80)
        
        if all_ok and embeddings_ok:
            print("\n✓ ALL 4 FIELDS WORKING CORRECTLY")
            print("\n  The implementation is complete:")
            print("    1. feedback_type: WORKING")
            print("    2. improvement_opportunity_type: WORKING")
            print("    3. classification_ar: WORKING")
            print("    4. classification_en: WORKING")
            print("    + Embeddings: AUTO-GENERATED")
            return True
        elif all_ok:
            print("\n! FIELDS OK but embeddings not generated")
            print("  Check if text fields are being sent correctly")
            return False
        else:
            print("\n✗ Some fields not populated")
            return False
            
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_all_four_fields()
    sys.exit(0 if success else 1)
