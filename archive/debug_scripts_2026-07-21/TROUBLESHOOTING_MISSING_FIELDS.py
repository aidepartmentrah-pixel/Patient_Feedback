"""
TROUBLESHOOTING: MISSING FIELDS IN ML DATABASE INSERT
======================================================
Issue: User manually added record, but these fields are NOT populated:
  - id (AUTO_INCREMENT not working)
  - classification_ar (needed for training)
  - classification_en (needed for training)
  - feedback_type (needed)
  - improvement_opportunity_type (needed)
  - ALL embedding fields (critical!)

This script traces the data flow to find where fields are lost.
"""

import sqlite3
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from backend.ml_mapping.ml_insert_adapter import FIELD_MAPPING, DIRECT_FIELDS, KNOWN_COLUMNS

ML_DB_PATH = workspace_root / "models_directory" / "patient_feedback_ml.db"

def check_database_schema():
    """Check if ID column is set to AUTOINCREMENT."""
    print("\n" + "="*80)
    print("ISSUE 1: AUTO_INCREMENT ID")
    print("="*80)
    
    try:
        conn = sqlite3.connect(str(ML_DB_PATH))
        cursor = conn.cursor()
        
        # Check table schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='patient_feedback_encoded'")
        schema = cursor.fetchone()
        
        if schema:
            print("\nDatabase Schema:")
            print(schema[0])
            
            if "AUTOINCREMENT" in schema[0]:
                print("\n[OK] AUTOINCREMENT is defined")
            else:
                print("\n[ERROR] AUTOINCREMENT is NOT defined!")
                print("  Solution: Recreate table with AUTOINCREMENT or use:")
                print("    INSERT INTO table (...) VALUES (...) WITHOUT rowid")
        
        conn.close()
    except Exception as e:
        print(f"[ERROR] Cannot check schema: {e}")

def check_field_mappings():
    """Check if all needed fields are in FIELD_MAPPING."""
    print("\n" + "="*80)
    print("ISSUE 2: MISSING FIELDS FROM MAPPINGS")
    print("="*80)
    
    needed_fields = {
        "classification_ar_id": "classification_ar (for training)",
        "classification_en_id": "classification_en (for training)",
        "feedback_type_id": "feedback_type (type mapping)",
        "improvement_opportunity_type_id": "improvement_opportunity_type (type mapping)",
    }
    
    print("\nField Mapping Configuration:")
    print(f"\nFIELD_MAPPING (mapped fields):")
    for input_key, (ml_col, entity_type) in FIELD_MAPPING.items():
        print(f"  {input_key} → {ml_col}")
    
    print(f"\nDIRECT_FIELDS (direct pass-through):")
    for input_key, ml_col in DIRECT_FIELDS.items():
        print(f"  {input_key} → {ml_col}")
    
    print(f"\nNeeded but missing from mappings:")
    for needed_key, description in needed_fields.items():
        in_field_mapping = needed_key in FIELD_MAPPING
        in_direct_fields = needed_key in DIRECT_FIELDS
        
        if in_field_mapping:
            print(f"  [OK] {needed_key} is in FIELD_MAPPING")
        elif in_direct_fields:
            print(f"  [OK] {needed_key} is in DIRECT_FIELDS")
        else:
            print(f"  [ERROR] {needed_key} ({description}) is NOT mapped!")
            print(f"         Need to add to either FIELD_MAPPING or DIRECT_FIELDS")

def check_embedding_field_mappings():
    """Check if embedding fields are in mappings."""
    print("\n" + "="*80)
    print("ISSUE 3: EMBEDDING FIELDS NOT MAPPED")
    print("="*80)
    
    embedding_fields = [
        "embedding_text1", "embedding_text2", "embedding_text3",
        "embedding_text123", "embedding_text23",
        "sentence_1_embedding", "sentence_2_embedding", "sentence_3_embedding",
        "sentence_4_embedding", "sentence_5_embedding", "sentence_6_embedding"
    ]
    
    print(f"\nEmbedding fields in DIRECT_FIELDS: {len([f for f in embedding_fields if f in DIRECT_FIELDS])}/{len(embedding_fields)}")
    
    for field in embedding_fields:
        if field in DIRECT_FIELDS:
            print(f"  [OK] {field}")
        else:
            print(f"  [ERROR] {field} NOT in DIRECT_FIELDS")
    
    print("\nNote: Embedding fields should be populated by _compute_text_embeddings() wrapper")
    print("      They are added to data dict BEFORE insertion")

def check_insert_function_signature():
    """Check what data the insert function expects."""
    print("\n" + "="*80)
    print("ISSUE 4: INSERT FUNCTION DATA FLOW")
    print("="*80)
    
    print("\nData Flow:")
    print("  1. UI/API → create_record() endpoint")
    print("  2. create_record() → add_corrected_record_to_ml(data)")
    print("  3. add_corrected_record_to_ml() → _compute_text_embeddings(data)")
    print("  4. _compute_text_embeddings() → enriches data with embeddings")
    print("  5. Enriched data → add_to_ml_database(enriched_data)")
    print("  6. add_to_ml_database() → inserts into DB")
    
    print("\nExpected input data fields:")
    print("  Text fields (for embedding generation):")
    print("    - complaint_text")
    print("    - immediate_action")
    print("    - taken_action")
    print("  Classification (from model predictions):")
    print("    - domain_id, category_id, subcategory_id")
    print("    - severity_id, stage_id, harm_id")
    print("  Type fields (MISSING!):")
    print("    - feedback_type")
    print("    - improvement_opportunity_type")
    print("    - classification_ar")
    print("    - classification_en")

def show_known_columns():
    """Show what columns the insert function knows about."""
    print("\n" + "="*80)
    print("ISSUE 5: KNOWN COLUMNS IN INSERT ADAPTER")
    print("="*80)
    
    print(f"\nKNOWN_COLUMNS ({len(KNOWN_COLUMNS)} columns):")
    for col in sorted(KNOWN_COLUMNS):
        in_direct = col in DIRECT_FIELDS
        in_field = any(v[0] == col for v in FIELD_MAPPING.values())
        
        status = ""
        if in_direct:
            status = "[DIRECT]"
        elif in_field:
            status = "[MAPPED]"
        else:
            status = "[UNKNOWN]"
        
        print(f"  {status} {col}")

def main():
    print("\n" + "="*80)
    print("TROUBLESHOOTING: MISSING FIELDS IN ML INSERT")
    print("="*80)
    
    check_database_schema()
    check_field_mappings()
    check_embedding_field_mappings()
    check_insert_function_signature()
    show_known_columns()
    
    print("\n" + "="*80)
    print("SUMMARY & SOLUTIONS")
    print("="*80)
    
    print("""
1. AUTO_INCREMENT ID:
   - Check if table has AUTOINCREMENT keyword
   - Solution: Recreate table or use INSERT with explicit NULL for id
   
2. Missing Fields (Not in mappings):
   - classification_ar, classification_en
   - feedback_type
   - improvement_opportunity_type
   
   Solution: Add these to FIELD_MAPPING or DIRECT_FIELDS
   
3. Embedding Fields Not Populated:
   - The wrapper function add_corrected_record_to_ml() should populate them
   - Check if wrapper is being called from create_record() endpoint
   - Verify _compute_text_embeddings() is working
   
4. Data Source Issue:
   - UI/API might not be sending all required fields
   - Check what data create_record() endpoint receives
   - Add logging to trace data through the pipeline

Next Steps:
  1. Check which endpoint is being used to add records
  2. Verify the endpoint is calling add_corrected_record_to_ml()
  3. Add logging to wrapper to see what data is received
  4. Update FIELD_MAPPING to include missing classification fields
""")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
