"""
TROUBLESHOOTING: ML INSERT END-TO-END TEST
==========================================
Input:  Data from UI (insert_record endpoint)
Output: Data in ML database
Goal:   Verify all columns are populated correctly
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import sys

# Add workspace root to path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.ml_mapping import add_corrected_record_to_ml

# ML Database path
ML_DB_PATH = workspace_root / "models_directory" / "patient_feedback_ml.db"

# Expected columns in ML database
EXPECTED_COLUMNS = {
    "id",
    "feedback_received_date",
    "feedback_type",
    "domain",
    "category",
    "sub_category",
    "classification_ar",
    "classification_en",
    "complaint_text",
    "immediate_action",
    "taken_action",
    "severity_level",
    "stage",
    "harm_level",
    "improvement_opportunity_type",
    "embedding_text1",
    "embedding_text2",
    "embedding_text3",
    "embedding_text123",
    "embedding_text23",
    "sentence_1_embedding",
    "sentence_2_embedding",
    "sentence_3_embedding",
    "sentence_4_embedding",
    "sentence_5_embedding",
    "sentence_6_embedding",
}

def get_db_columns():
    """Get actual columns from database schema."""
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(patient_feedback_encoded)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()
        return columns
    except Exception as e:
        print(f"[ERROR] Cannot read database schema: {e}")
        return set()

def create_test_input():
    """Create realistic UI input data."""
    return {
        # Text fields (from user correction)
        "complaint_text": "المريض يشكو من ألم شديد في البطن وتأخر في تقديم العلاج الطبي",
        "immediate_action": "تم إعطاء المريض مسكنات قوية والتحقق من الحالة",
        "taken_action": "تم نقل المريض إلى قسم الطوارئ لمتابعة دقيقة",
        
        # Classification (from model prediction)
        "domain_id": 1,
        "category_id": 5,
        "subcategory_id": 5,
        "severity_id": 1,
        "stage_id": 1,
        "harm_id": 5,
        
        # Patient info
        "feedback_received_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        
        # Classification details
        "classification_ar": 5.0,
        "classification_en": 5,
        "feedback_type": 1,
        "improvement_opportunity_type": 2,
    }

def test_ml_insert():
    """Test the full ML insert pipeline."""
    print("\n" + "="*80)
    print("TROUBLESHOOTING: ML INSERT END-TO-END TEST")
    print("="*80)
    
    # Step 1: Check database schema
    print("\n[STEP 1] Checking Database Schema...")
    db_columns = get_db_columns()
    if not db_columns:
        print("[ERROR] Cannot access ML database")
        return
    
    print(f"  Database columns found: {len(db_columns)}")
    print(f"  Expected columns: {len(EXPECTED_COLUMNS)}")
    
    missing_in_db = EXPECTED_COLUMNS - db_columns
    extra_in_db = db_columns - EXPECTED_COLUMNS
    
    if missing_in_db:
        print(f"\n  [WARNING] Missing columns in database:")
        for col in sorted(missing_in_db):
            print(f"    - {col}")
    
    if extra_in_db:
        print(f"\n  [INFO] Extra columns in database:")
        for col in sorted(extra_in_db):
            print(f"    - {col}")
    
    # Step 2: Prepare test data
    print("\n[STEP 2] Creating Test Input Data...")
    test_data = create_test_input()
    print(f"  Input fields: {len(test_data)}")
    for key, value in test_data.items():
        value_preview = str(value)[:50] if len(str(value)) > 50 else str(value)
        print(f"    [OK] {key}: {value_preview}")
    
    # Step 3: Get record count before insert
    print("\n[STEP 3] Getting Record Count Before Insert...")
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patient_feedback_encoded")
        count_before = cursor.fetchone()[0]
        print(f"  Records before: {count_before}")
        conn.close()
    except Exception as e:
        print(f"  [ERROR] Cannot count records: {e}")
        return
    
    # Step 4: Insert data
    print("\n[STEP 4] Inserting Data into ML Database...")
    try:
        add_corrected_record_to_ml(test_data)
        print("  [OK] Insert function executed")
    except Exception as e:
        print(f"  [ERROR] Insert failed: {e}")
        return
    
    # Step 5: Verify insert
    print("\n[STEP 5] Verifying Insert...")
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM patient_feedback_encoded")
        count_after = cursor.fetchone()[0]
        print(f"  Records after: {count_after}")
        
        if count_after > count_before:
            print(f"  [OK] Record inserted ({count_after - count_before} new records)")
        else:
            print(f"  [WARNING] No new records inserted")
            conn.close()
            return
        
        # Step 6: Check the newly inserted record
        print("\n[STEP 6] Checking Newly Inserted Record...")
        cursor.execute("""
            SELECT * FROM patient_feedback_encoded 
            ORDER BY id DESC LIMIT 1
        """)
        record = cursor.fetchone()
        
        if not record:
            print("  [ERROR] Cannot retrieve inserted record")
            conn.close()
            return
        
        # Get column names
        cursor.execute("PRAGMA table_info(patient_feedback_encoded)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Create record dict
        record_dict = dict(zip(columns, record))
        
        # Step 7: Analyze populated columns
        print("\n[STEP 7] Column Population Analysis:")
        populated = 0
        empty = 0
        null = 0
        
        for col in sorted(columns):
            value = record_dict.get(col)
            
            if value is None:
                print(f"  [NULL] {col}: None")
                null += 1
            elif value == "" or value == 0:
                print(f"  [EMPTY] {col}: {repr(value)}")
                empty += 1
            else:
                value_preview = str(value)[:60] if len(str(value)) > 60 else str(value)
                print(f"  [OK] {col}: {value_preview}")
                populated += 1
        
        # Summary
        print(f"\n  SUMMARY:")
        print(f"    Populated: {populated}")
        print(f"    Empty: {empty}")
        print(f"    NULL: {null}")
        print(f"    Total: {populated + empty + null}")
        
        # Step 8: Check embeddings
        print("\n[STEP 8] Embedding Check:")
        embedding_fields = [
            "embedding_text1", "embedding_text2", "embedding_text3",
            "embedding_text123", "embedding_text23",
            "sentence_1_embedding", "sentence_2_embedding", "sentence_3_embedding",
            "sentence_4_embedding", "sentence_5_embedding", "sentence_6_embedding"
        ]
        
        for field in embedding_fields:
            value = record_dict.get(field)
            if value is None:
                print(f"  [NULL] {field}")
            elif isinstance(value, bytes):
                print(f"  [OK] {field}: {len(value)} bytes")
            elif value == "":
                print(f"  [EMPTY] {field}")
            else:
                print(f"  [?] {field}: {type(value).__name__} = {str(value)[:50]}")
        
        conn.close()
        
        # Step 9: Final report
        print("\n" + "="*80)
        print("TROUBLESHOOTING REPORT")
        print("="*80)
        
        issues = []
        
        if null > 5:
            issues.append(f"Too many NULL columns ({null})")
        
        if not record_dict.get("complaint_text"):
            issues.append("complaint_text not populated")
        
        embedding_count = sum(1 for f in embedding_fields if record_dict.get(f))
        if embedding_count < 5:
            issues.append(f"Only {embedding_count}/{len(embedding_fields)} embeddings generated")
        
        if not issues:
            print("\n[OK] ALL CHECKS PASSED - ML INSERT WORKING CORRECTLY")
            print(f"\n  Inserted record has:")
            print(f"    - {populated} populated columns")
            print(f"    - {embedding_count} embedding fields")
            print(f"    - {empty} empty columns")
        else:
            print("\n[WARNING] ISSUES FOUND - TROUBLESHOOTING REQUIRED:")
            for i, issue in enumerate(issues, 1):
                print(f"\n  {i}. {issue}")
        
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_insert()
