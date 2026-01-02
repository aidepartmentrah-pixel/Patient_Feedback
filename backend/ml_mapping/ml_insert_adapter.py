"""
ML Insert Adapter Module

Translates database values → ML values and inserts rows into patient_feedback_encoded.

Acts as a bridge between business DB and ML training data.
Silent, boring, predictable, and safe.

No training logic, no feature engineering, no schema changes.
"""

import sqlite3
import os
from typing import Dict, List, Optional, Any
from itertools import product

# Import the mapping helper
from backend.config.ml_encoding_mapper import map_db_to_ml

# Import embedding functions from modular_functions
try:
    from models_directory.Classification_Models.Stage.modular_functions import (
        split_arabic_text_into_sentences,
        get_embedding,
        get_embedding_list,
        l2_normalize
    )
    _EMBEDDING_FUNCTIONS_AVAILABLE = True
except ImportError:
    _EMBEDDING_FUNCTIONS_AVAILABLE = False
    print("[WARNING] Embedding functions not available - embeddings will be skipped")


# Database configuration
ML_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models_directory",
    "patient_feedback_ml.db"
)

# Known columns in patient_feedback_encoded table
KNOWN_COLUMNS = {
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

# Mapping from input data keys to ML table column names
FIELD_MAPPING = {
    "domain_id": ("domain", "domain"),
    "category_id": ("category", "category"),
    "subcategory_id": ("sub_category", "subcategory"),
    "severity_id": ("severity_level", "severity_level"),
    "stage_id": ("stage", "stage"),
    "harm_id": ("harm_level", "harm_level"),
}

# Direct pass-through fields (no mapping needed)
DIRECT_FIELDS = {
    "feedback_received_date": "feedback_received_date",
    "feedback_type": "feedback_type",
    "classification_ar": "classification_ar",
    "classification_en": "classification_en",
    "complaint_text": "complaint_text",
    "immediate_action": "immediate_action",
    "taken_action": "taken_action",
    "improvement_opportunity_type": "improvement_opportunity_type",
    "embedding_text1": "embedding_text1",
    "embedding_text2": "embedding_text2",
    "embedding_text3": "embedding_text3",
    "embedding_text123": "embedding_text123",
    "embedding_text23": "embedding_text23",
    "sentence_1_embedding": "sentence_1_embedding",
    "sentence_2_embedding": "sentence_2_embedding",
    "sentence_3_embedding": "sentence_3_embedding",
    "sentence_4_embedding": "sentence_4_embedding",
    "sentence_5_embedding": "sentence_5_embedding",
    "sentence_6_embedding": "sentence_6_embedding",
}


def _get_mapped_values(data: Dict[str, Any]) -> Dict[str, List[int]]:
    """
    Extract mapped field values from input data.
    
    Returns a dict where each key has a list of ML IDs.
    If mapping is empty, that key is excluded.
    
    Args:
        data: Input data dictionary
    
    Returns:
        Dict with mapped field names → list of ML IDs
    """
    mapped_values = {}
    
    for input_key, (ml_column, entity_type) in FIELD_MAPPING.items():
        if input_key not in data or data[input_key] is None:
            continue
        
        try:
            db_id = int(data[input_key])
            ml_ids = map_db_to_ml(entity_type, db_id)
            
            if ml_ids:  # Only include non-empty mappings
                mapped_values[ml_column] = ml_ids
        except (ValueError, TypeError):
            # Cannot convert to int, skip this field
            pass
        except Exception:
            # Mapping failed, skip
            pass
    
    return mapped_values


def _get_direct_values(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract direct pass-through field values from input data.
    
    Args:
        data: Input data dictionary
    
    Returns:
        Dict with column names → values (only non-None values)
    """
    direct_values = {}
    
    for input_key, ml_column in DIRECT_FIELDS.items():
        if input_key in data and data[input_key] is not None:
            direct_values[ml_column] = data[input_key]
    
    return direct_values


def _generate_combinations(mapped_values: Dict[str, List[int]]) -> List[Dict[str, int]]:
    """
    Generate all combinations of mapped field values.
    
    If domain=[1], category=[4,6], this generates:
    [{'domain': 1, 'category': 4}, {'domain': 1, 'category': 6}]
    
    If no mappings, returns [{}] (empty dict for single row).
    
    Args:
        mapped_values: Dict of field → list of IDs
    
    Returns:
        List of dicts, each representing one row to insert
    """
    if not mapped_values:
        return [{}]
    
    # Get all keys and their value lists
    keys = list(mapped_values.keys())
    value_lists = [mapped_values[k] for k in keys]
    
    # Generate all combinations
    combinations = []
    for value_combo in product(*value_lists):
        combo_dict = dict(zip(keys, value_combo))
        combinations.append(combo_dict)
    
    return combinations


def _insert_row(db_cursor: sqlite3.Cursor, row_data: Dict[str, Any]) -> bool:
    """
    Insert a single row into patient_feedback_encoded.
    
    Handles auto-increment ID generation since SQLite table lacks AUTOINCREMENT.
    
    Args:
        db_cursor: SQLite cursor
        row_data: Data to insert (column name → value)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Filter to only known columns
        filtered_data = {k: v for k, v in row_data.items() if k in KNOWN_COLUMNS}
        
        if not filtered_data:
            return False
        
        # Handle auto-increment ID
        if "id" not in filtered_data or filtered_data["id"] is None:
            try:
                # Get next available ID
                db_cursor.execute("SELECT MAX(id) FROM patient_feedback_encoded")
                max_id = db_cursor.fetchone()[0] or 0
                filtered_data["id"] = max_id + 1
            except:
                # If query fails, let database handle it
                pass
        
        # Build INSERT statement
        columns = list(filtered_data.keys())
        placeholders = ",".join(["?" for _ in columns])
        column_names = ",".join(columns)
        
        sql = f"INSERT INTO patient_feedback_encoded ({column_names}) VALUES ({placeholders})"
        values = [filtered_data[col] for col in columns]
        
        db_cursor.execute(sql, values)
        return True
    except Exception as e:
        print(f"Warning: Failed to insert row into ML database: {str(e)}")
        return False


def _compute_text_embeddings(data: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    """
    Generate embeddings for complaint_text, immediate_action, taken_action, and combinations.
    
    Extracts text fields, generates embeddings, and returns enrichment dict.
    
    Args:
        data: Input data dict (must contain text fields)
    
    Returns:
        Dict with embedding fields → embedding bytes (or None if generation fails)
    """
    if not _EMBEDDING_FUNCTIONS_AVAILABLE:
        return {}
    
    embeddings = {}
    
    try:
        # Extract text fields (safe defaults to empty string)
        complaint_text = data.get('complaint_text') or ""
        immediate_action = data.get('immediate_action') or ""
        taken_action = data.get('taken_action') or ""
        
        # ====================================================
        # 1. Individual field embeddings
        # ====================================================
        try:
            if complaint_text.strip():
                embeddings['embedding_text1'] = get_embedding(complaint_text)
            else:
                embeddings['embedding_text1'] = None
        except Exception as e:
            print(f"[Embedding Warning] complaint_text embedding failed: {str(e)}")
            embeddings['embedding_text1'] = None
        
        try:
            if immediate_action.strip():
                embeddings['embedding_text2'] = get_embedding(immediate_action)
            else:
                embeddings['embedding_text2'] = None
        except Exception as e:
            print(f"[Embedding Warning] immediate_action embedding failed: {str(e)}")
            embeddings['embedding_text2'] = None
        
        try:
            if taken_action.strip():
                embeddings['embedding_text3'] = get_embedding(taken_action)
            else:
                embeddings['embedding_text3'] = None
        except Exception as e:
            print(f"[Embedding Warning] taken_action embedding failed: {str(e)}")
            embeddings['embedding_text3'] = None
        
        # ====================================================
        # 2. Combination embeddings
        # ====================================================
        try:
            text123 = f"{complaint_text} {immediate_action} {taken_action}".strip()
            if text123:
                embeddings['embedding_text123'] = get_embedding(text123)
            else:
                embeddings['embedding_text123'] = None
        except Exception as e:
            print(f"[Embedding Warning] text123 embedding failed: {str(e)}")
            embeddings['embedding_text123'] = None
        
        try:
            text23 = f"{immediate_action} {taken_action}".strip()
            if text23:
                embeddings['embedding_text23'] = get_embedding(text23)
            else:
                embeddings['embedding_text23'] = None
        except Exception as e:
            print(f"[Embedding Warning] text23 embedding failed: {str(e)}")
            embeddings['embedding_text23'] = None
        
        # ====================================================
        # 3. Sentence embeddings (split complaint_text into 6 sentences)
        # ====================================================
        try:
            if complaint_text.strip():
                sentences = split_arabic_text_into_sentences(complaint_text, max_sentences=6)
                
                # Get embeddings for all sentences in batch
                if sentences:
                    sentence_embeddings = get_embedding_list(sentences)
                    
                    # Assign to sentence_X_embedding fields
                    for i in range(6):
                        if i < len(sentence_embeddings):
                            embeddings[f'sentence_{i+1}_embedding'] = sentence_embeddings[i]
                        else:
                            embeddings[f'sentence_{i+1}_embedding'] = None
                else:
                    # No sentences extracted
                    for i in range(6):
                        embeddings[f'sentence_{i+1}_embedding'] = None
            else:
                # Empty complaint_text
                for i in range(6):
                    embeddings[f'sentence_{i+1}_embedding'] = None
        except Exception as e:
            print(f"[Embedding Warning] sentence embeddings failed: {str(e)}")
            for i in range(6):
                embeddings[f'sentence_{i+1}_embedding'] = None
    
    except Exception as e:
        print(f"[Embedding Warning] Unexpected error in embedding computation: {str(e)}")
        # Return empty dict - will skip embeddings for this record
        return {}
    
    return embeddings


def add_corrected_record_to_ml(data: dict) -> None:
    """
    PUBLIC WRAPPER - Insert human-corrected record to ML database WITH embeddings.
    
    This is the interface entry point for receiving corrected data from the UI.
    
    Process:
    1. Generates embeddings for complaint_text, immediate_action, taken_action
    2. Splits complaint_text into sentences and embeds each
    3. Creates combination embeddings (text123, text23)
    4. Enriches data dict with all embeddings
    5. Calls add_to_ml_database() with enriched data
    
    Args:
        data (dict): User-corrected data from Interface
                    Should contain: record_id, complaint_text, immediate_action,
                                   taken_action, domain_id, category_id, etc.
    
    Returns:
        None
    
    Note:
        - Never raises exceptions
        - Gracefully handles missing embeddings functions
        - Logs warnings using print() only
        - Embedding generation is optional - record inserts even if embeddings fail
    """
    if not data:
        return
    
    try:
        # Make a copy to avoid modifying original
        enriched_data = dict(data)
        
        # Generate embeddings (gracefully fails if functions unavailable)
        embedding_dict = _compute_text_embeddings(data)
        
        # Merge embeddings into data
        enriched_data.update(embedding_dict)
        
        # Call the internal insert function with enriched data
        add_to_ml_database(enriched_data)
    
    except Exception as e:
        print(f"[ML INSERT WARNING] Unexpected error in add_corrected_record_to_ml: {str(e)}")


def add_to_ml_database(data: dict) -> None:
    """
    Insert a record into the ML database.
    
    Translates database values → ML values using Database_To_ML_Encoding.json
    and inserts into patient_feedback_encoded table.
    
    Handles one-to-many mappings by inserting multiple rows if needed.
    
    Args:
        data (dict): Input data from create_record()
                    Should contain keys like record_id, domain_id, category_id, etc.
    
    Returns:
        None
    
    Note:
        - Never raises exceptions
        - Logs warnings using print() only
        - Silently skips missing or invalid data
        - Only inserts known columns
    """
    if not data:
        return
    
    try:
        # Extract mapped values (with transformations)
        mapped_values = _get_mapped_values(data)
        
        # Extract direct values (pass-through)
        direct_values = _get_direct_values(data)
        
        # Generate all row combinations
        combinations = _generate_combinations(mapped_values)
        
        # Connect to ML database
        try:
            connection = sqlite3.connect(ML_DB_PATH)
            cursor = connection.cursor()
        except Exception as e:
            print(f"Warning: Could not connect to ML database at {ML_DB_PATH}: {str(e)}")
            return
        
        try:
            # Insert each combination
            inserted_count = 0
            for combo in combinations:
                # Merge mapped and direct values
                row_data = {**direct_values, **combo}
                
                if _insert_row(cursor, row_data):
                    inserted_count += 1
            
            # Commit all inserts
            if inserted_count > 0:
                connection.commit()
        
        except Exception as e:
            print(f"Warning: Error during ML database insert: {str(e)}")
            connection.rollback()
        
        finally:
            connection.close()
    
    except Exception as e:
        print(f"Warning: Unexpected error in add_to_ml_database: {str(e)}")
