"""
ML Encoding Mapper Module

Responsible for loading and interpreting the Database_To_ML_Encoding.json file.
Provides lazy-loaded, cached access to ML ID mappings.

No database access, no logging, safe defaults only.
"""

import json
import os
from typing import Dict, List

# Global cache for the mapping
_mapping_cache = None
_mapping_file_path = None


def _get_mapping_file_path() -> str:
    """Determine the path to Database_To_ML_Encoding.json"""
    global _mapping_file_path
    
    if _mapping_file_path is not None:
        return _mapping_file_path
    
    # Get the directory of this file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    _mapping_file_path = os.path.join(current_dir, "Database_To_ML_Encoding.json")
    
    return _mapping_file_path


def load_mapping() -> dict:
    """
    Load the ML mapping from JSON file.
    Uses lazy loading - loads only once and caches in memory.
    
    Returns:
        dict: Parsed mapping dictionary from Database_To_ML_Encoding.json
    """
    global _mapping_cache
    
    # Return cached mapping if already loaded
    if _mapping_cache is not None:
        return _mapping_cache
    
    try:
        file_path = _get_mapping_file_path()
        with open(file_path, 'r', encoding='utf-8') as f:
            _mapping_cache = json.load(f)
        return _mapping_cache
    except (FileNotFoundError, json.JSONDecodeError):
        # Return empty dict with safe defaults
        _mapping_cache = {"reverseIdMap": {}}
        return _mapping_cache


def map_db_to_ml(entity: str, db_id: int) -> List[int]:
    """
    Map a database ID to ML model IDs using the reverseIdMap.
    
    Args:
        entity (str): Entity type (e.g., 'domain', 'category', 'subcategory')
        db_id (int): Database ID to map
    
    Returns:
        List[int]: List of ML IDs (empty list if mapping not found)
    """
    try:
        mapping = load_mapping()
        reverse_map = mapping.get("reverseIdMap", {})
        entity_map = reverse_map.get(entity, {})
        
        # Convert db_id to string key (JSON keys are strings)
        result = entity_map.get(str(db_id), [])
        
        # Ensure we always return a list
        if isinstance(result, list):
            return result
        else:
            return []
    except Exception:
        # Never raise an exception, return empty list
        return []


def get_all_mapped_fields() -> List[str]:
    """
    Get list of available entity types in the mapping.
    
    Returns:
        List[str]: List of entity keys (e.g., ['domain', 'category', 'subcategory'])
    """
    try:
        mapping = load_mapping()
        reverse_map = mapping.get("reverseIdMap", {})
        return list(reverse_map.keys())
    except Exception:
        # Never raise an exception, return empty list
        return []
