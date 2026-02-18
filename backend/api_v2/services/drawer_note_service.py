"""
Drawer Note Service Layer - Phase G-B5

This module contains BUSINESS LOGIC for Drawer Notes.
- Orchestrates DB layer calls
- Enforces validation rules
- NO SQL queries (uses DB layer only)

Business Rules (FROZEN):
- Notes are editable (overwrite allowed)
- Notes use soft delete only (IsDeleted = 1)
- Notes must always have at least ONE label
- Only ACTIVE labels allowed
- Cannot modify deleted notes
- Empty text not allowed (after trim)

Author: Phase G Implementation
"""

from api_v2.db_layer import drawer_note_db
from api_v2.db_layer import drawer_label_db


def create_note_with_labels(note_text, label_ids, created_by_user_id, created_by_name):
    """
    Create a new drawer note with labels.
    
    Business Rules:
    - Text cannot be empty (after trim)
    - Must have at least one label
    - All labels must exist and be active
    
    Args:
        note_text (str): The note content
        label_ids (list): List of label IDs to attach
        created_by_user_id (int): User ID creating the note
        created_by_name (str): User name creating the note
    
    Returns:
        int: The created note ID
    
    Raises:
        ValueError: If validation fails
    """
    # Trim and validate text
    trimmed_text = note_text.strip() if note_text else ""
    if not trimmed_text:
        raise ValueError("Note text cannot be empty")
    
    # Validate labels not empty
    if not label_ids or len(label_ids) == 0:
        raise ValueError("Note must have at least one label")
    
    # Validate all labels exist and are active
    valid_label_ids = drawer_label_db.get_label_ids_exist(label_ids)
    if len(valid_label_ids) != len(label_ids):
        invalid_ids = set(label_ids) - set(valid_label_ids)
        raise ValueError(f"Invalid or inactive label IDs: {invalid_ids}")
    
    # Insert note
    note_id = drawer_note_db.insert_note(
        note_text=trimmed_text,
        created_by_user_id=created_by_user_id,
        created_by_name=created_by_name
    )
    
    # Attach labels
    drawer_note_db.attach_labels_to_note(note_id, label_ids)
    
    return note_id


def edit_note_text(note_id, new_text):
    """
    Edit the text content of an existing note.
    
    Business Rules:
    - Text cannot be empty (after trim)
    - Note must exist
    - Note cannot be deleted
    
    Args:
        note_id (int): The note ID to edit
        new_text (str): The new note content
    
    Raises:
        ValueError: If validation fails or note is deleted
    """
    # Trim and validate text
    trimmed_text = new_text.strip() if new_text else ""
    if not trimmed_text:
        raise ValueError("Note text cannot be empty")
    
    # Load note
    note = drawer_note_db.get_note_by_id(note_id)
    if not note:
        raise ValueError(f"Note {note_id} not found")
    
    # Check not deleted
    if note.get('is_deleted', False):
        raise ValueError(f"Cannot edit deleted note {note_id}")
    
    # Update text
    drawer_note_db.update_note_text(note_id, trimmed_text)


def edit_note_labels(note_id, label_ids):
    """
    Replace the labels attached to a note.
    
    Business Rules:
    - Must have at least one label
    - All labels must exist and be active
    - Note must exist
    - Note cannot be deleted
    
    Args:
        note_id (int): The note ID to edit
        label_ids (list): New list of label IDs
    
    Raises:
        ValueError: If validation fails
    """
    # Validate labels not empty
    if not label_ids or len(label_ids) == 0:
        raise ValueError("Note must have at least one label")
    
    # Validate all labels exist and are active
    valid_label_ids = drawer_label_db.get_label_ids_exist(label_ids)
    if len(valid_label_ids) != len(label_ids):
        invalid_ids = set(label_ids) - set(valid_label_ids)
        raise ValueError(f"Invalid or inactive label IDs: {invalid_ids}")
    
    # Load note
    note = drawer_note_db.get_note_by_id(note_id)
    if not note:
        raise ValueError(f"Note {note_id} not found")
    
    # Check not deleted
    if note.get('is_deleted', False):
        raise ValueError(f"Cannot edit deleted note {note_id}")
    
    # Replace labels
    drawer_note_db.replace_note_labels(note_id, label_ids)


def soft_delete_note(note_id):
    """
    Soft delete a note (sets IsDeleted = 1).
    
    Business Rules:
    - Note must exist
    
    Args:
        note_id (int): The note ID to delete
    
    Raises:
        ValueError: If note not found
    """
    # Check note exists
    note = drawer_note_db.get_note_by_id(note_id)
    if not note:
        raise ValueError(f"Note {note_id} not found")
    
    # Soft delete
    drawer_note_db.soft_delete_note(note_id)


def get_note_detail(note_id):
    """
    Get full note details including attached labels.
    
    Args:
        note_id (int): The note ID to retrieve
    
    Returns:
        dict: Note data with 'label_ids' key added, or None if not found
    """
    # Load note
    note = drawer_note_db.get_note_by_id(note_id)
    if not note:
        return None
    
    # Load label IDs
    label_ids = drawer_note_db.get_note_label_ids(note_id)
    
    # Combine
    note['label_ids'] = label_ids
    
    return note


def list_notes(label_ids=None, limit=50, offset=0):
    """
    List notes with optional label filtering.
    
    Business Rules:
    - Only returns non-deleted notes
    - If label_ids provided, uses AND filtering (must have ALL labels)
    
    Args:
        label_ids (list, optional): Filter by labels (AND logic). Defaults to None.
        limit (int, optional): Max results. Defaults to 50.
        offset (int, optional): Pagination offset. Defaults to 0.
    
    Returns:
        list: List of note dicts
    """
    if label_ids and len(label_ids) > 0:
        # Filter by labels (AND logic)
        return drawer_note_db.filter_notes_by_label_ids(label_ids, limit, offset)
    else:
        # List all active notes
        return drawer_note_db.list_notes_paged(limit, offset)
