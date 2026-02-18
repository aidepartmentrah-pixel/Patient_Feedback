"""
Drawer Label Service Layer - Phase G-B6

This module contains BUSINESS LOGIC for Drawer Labels.
- Orchestrates DB layer calls
- Enforces validation rules
- NO SQL queries (uses DB layer only)

Business Rules (FROZEN):
- label_name required
- Trimmed before validation
- Min length: 2 characters
- Max length: 100 characters
- Must be unique (DB constraint enforces)
- Disable = soft disable only (IsActive = 0)
- Disabled labels cannot be used by note service

Author: Phase G Implementation
"""

from api_v2.db_layer import drawer_label_db


def create_label(label_name):
    """
    Create a new drawer label.
    
    Business Rules:
    - Label name is trimmed
    - Must be at least 2 characters
    - Must be at most 100 characters
    - Must be unique (enforced by DB constraint)
    
    Args:
        label_name (str): The label name
    
    Returns:
        int: The created label ID
    
    Raises:
        ValueError: If validation fails
    """
    # Trim and validate
    trimmed_name = label_name.strip() if label_name else ""
    
    if len(trimmed_name) < 2:
        raise ValueError("Label name must be at least 2 characters")
    
    if len(trimmed_name) > 100:
        raise ValueError("Label name must be at most 100 characters")
    
    # Insert label (DB will enforce uniqueness)
    label_id = drawer_label_db.insert_label(trimmed_name)
    
    return label_id


def list_active_labels():
    """
    Get all active labels.
    
    Returns:
        list: List of active label dicts with keys: label_id, label_name, is_active, created_at
    """
    return drawer_label_db.list_active_labels()


def disable_label(label_id):
    """
    Disable a label (sets IsActive = 0).
    
    This is a soft disable - label still exists in database but cannot be used.
    Existing note-label links remain intact.
    
    Args:
        label_id (int): The label ID to disable
    """
    drawer_label_db.disable_label(label_id)


def validate_label_ids_active(label_ids):
    """
    Validate that all provided label IDs exist and are active.
    
    Used by note service to ensure only active labels can be attached to notes.
    
    Args:
        label_ids (list): List of label IDs to validate
    
    Raises:
        ValueError: If any label ID is invalid or inactive
    """
    if not label_ids:
        return  # Empty list is valid (though note service requires at least one)
    
    # Get valid active IDs
    valid_ids = drawer_label_db.get_label_ids_exist(label_ids)
    
    # Check if all provided IDs are valid
    if len(valid_ids) != len(label_ids):
        invalid_ids = set(label_ids) - set(valid_ids)
        raise ValueError(f"Invalid or inactive label IDs: {invalid_ids}")
