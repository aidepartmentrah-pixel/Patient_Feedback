"""
Drawer Note Database Layer (API V2 - Phase G)
Handles SQL operations for APP_DrawerNote and APP_DrawerNoteLabelLink tables.

This is part of Phase G Drawer Notes multi-label system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from core.database import get_connection


# ============================================================
# NOTE CRUD OPERATIONS
# ============================================================

def insert_note(
    note_text: str,
    created_by_user_id: int,
    created_by_name: str,
    patient_admission_id: Optional[int] = None
) -> int:
    """
    Create a new drawer note.
    
    Args:
        note_text: Note content
        created_by_user_id: User ID who created the note
        created_by_name: User name who created the note
        patient_admission_id: Optional patient admission ID to link note to a patient
    
    Returns:
        NoteID of newly created note
    
    Raises:
        Exception: If insert fails
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO dbo.APP_DrawerNote (
                NoteText,
                CreatedByUserID,
                CreatedByName,
                PatientAdmissionID
            )
            OUTPUT INSERTED.NoteID
            VALUES (?, ?, ?, ?)
        """
        
        cursor.execute(query, (note_text, created_by_user_id, created_by_name, patient_admission_id))
        note_id = cursor.fetchone()[0]
        conn.commit()
        
        return note_id
        
    finally:
        cursor.close()
        conn.close()


def update_note_text(note_id: int, new_text: str) -> None:
    """
    Update the text content of a drawer note.
    
    Args:
        note_id: ID of note to update
        new_text: New note content
    
    Raises:
        Exception: If update fails
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_DrawerNote
            SET NoteText = ?
            WHERE NoteID = ?
        """
        
        cursor.execute(query, (new_text, note_id))
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def soft_delete_note(note_id: int) -> None:
    """
    Soft delete a drawer note by setting IsDeleted = 1.
    
    Args:
        note_id: ID of note to soft delete
    
    Raises:
        Exception: If update fails
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_DrawerNote
            SET IsDeleted = 1
            WHERE NoteID = ?
        """
        
        cursor.execute(query, (note_id,))
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def get_note_by_id(note_id: int) -> Optional[Dict[str, Any]]:
    """
    Get a drawer note by ID.
    
    Args:
        note_id: ID of note to retrieve
    
    Returns:
        Dict with note data or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                n.NoteID,
                n.NoteText,
                n.CreatedAt,
                n.CreatedByUserID,
                n.CreatedByName,
                n.IsDeleted,
                n.PatientAdmissionID,
                COALESCE(p.FullName, r.FullName) as PatientName
            FROM dbo.APP_DrawerNote n
            LEFT JOIN dbo.VW_PatientAdmission p ON n.PatientAdmissionID = p.PatientAdmissionID
            LEFT JOIN dbo.APP_RESERVE_PATIENT r ON n.PatientAdmissionID = r.PatientAdmissionID AND p.PatientAdmissionID IS NULL
            WHERE n.NoteID = ?
        """
        
        cursor.execute(query, (note_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            'note_id': row.NoteID,
            'note_text': row.NoteText,
            'created_at': row.CreatedAt,
            'created_by_user_id': row.CreatedByUserID,
            'created_by_name': row.CreatedByName,
            'is_deleted': row.IsDeleted,
            'patient_admission_id': row.PatientAdmissionID,
            'patient_name': row.PatientName
        }
        
    finally:
        cursor.close()
        conn.close()


def list_notes_paged(limit: int, offset: int, patient_admission_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Get paginated list of non-deleted notes, ordered by created_at DESC.
    
    Args:
        limit: Maximum number of notes to return
        offset: Number of notes to skip
        patient_admission_id: Optional filter by patient
    
    Returns:
        List of note dicts
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        if patient_admission_id is not None:
            query = """
                SELECT 
                    n.NoteID,
                    n.NoteText,
                    n.CreatedAt,
                    n.CreatedByUserID,
                    n.CreatedByName,
                    n.IsDeleted,
                    n.PatientAdmissionID,
                    COALESCE(p.FullName, r.FullName) as PatientName
                FROM dbo.APP_DrawerNote n
                LEFT JOIN dbo.VW_PatientAdmission p ON n.PatientAdmissionID = p.PatientAdmissionID
                LEFT JOIN dbo.APP_RESERVE_PATIENT r ON n.PatientAdmissionID = r.PatientAdmissionID AND p.PatientAdmissionID IS NULL
                WHERE n.IsDeleted = 0 AND n.PatientAdmissionID = ?
                ORDER BY n.CreatedAt DESC
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY
            """
            cursor.execute(query, (patient_admission_id, offset, limit))
        else:
            query = """
                SELECT 
                    n.NoteID,
                    n.NoteText,
                    n.CreatedAt,
                    n.CreatedByUserID,
                    n.CreatedByName,
                    n.IsDeleted,
                    n.PatientAdmissionID,
                    COALESCE(p.FullName, r.FullName) as PatientName
                FROM dbo.APP_DrawerNote n
                LEFT JOIN dbo.VW_PatientAdmission p ON n.PatientAdmissionID = p.PatientAdmissionID
                LEFT JOIN dbo.APP_RESERVE_PATIENT r ON n.PatientAdmissionID = r.PatientAdmissionID AND p.PatientAdmissionID IS NULL
                WHERE n.IsDeleted = 0
                ORDER BY n.CreatedAt DESC
                OFFSET ? ROWS
                FETCH NEXT ? ROWS ONLY
            """
            cursor.execute(query, (offset, limit))
        
        rows = cursor.fetchall()
        
        notes = []
        for row in rows:
            notes.append({
                'note_id': row.NoteID,
                'note_text': row.NoteText,
                'created_at': row.CreatedAt,
                'created_by_user_id': row.CreatedByUserID,
                'created_by_name': row.CreatedByName,
                'is_deleted': row.IsDeleted,
                'patient_admission_id': row.PatientAdmissionID,
                'patient_name': row.PatientName
            })
        
        return notes
        
    finally:
        cursor.close()
        conn.close()


# ============================================================
# LABEL LINKING OPERATIONS
# ============================================================

def attach_labels_to_note(note_id: int, label_ids: List[int]) -> None:
    """
    Attach labels to a note (insert into link table).
    Does not remove existing labels - only adds new ones.
    
    Args:
        note_id: ID of note
        label_ids: List of label IDs to attach
    
    Raises:
        Exception: If insert fails (e.g., duplicate link)
    """
    if not label_ids:
        return
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO dbo.APP_DrawerNoteLabelLink (NoteID, LabelID)
            VALUES (?, ?)
        """
        
        for label_id in label_ids:
            cursor.execute(query, (note_id, label_id))
        
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def replace_note_labels(note_id: int, label_ids: List[int]) -> None:
    """
    Replace all labels for a note.
    Deletes existing label links, then inserts new ones.
    
    Args:
        note_id: ID of note
        label_ids: New list of label IDs (empty list removes all labels)
    
    Raises:
        Exception: If operation fails
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Delete existing links
        delete_query = """
            DELETE FROM dbo.APP_DrawerNoteLabelLink
            WHERE NoteID = ?
        """
        cursor.execute(delete_query, (note_id,))
        
        # Insert new links
        if label_ids:
            insert_query = """
                INSERT INTO dbo.APP_DrawerNoteLabelLink (NoteID, LabelID)
                VALUES (?, ?)
            """
            
            for label_id in label_ids:
                cursor.execute(insert_query, (note_id, label_id))
        
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def get_note_label_ids(note_id: int) -> List[int]:
    """
    Get all label IDs attached to a note.
    
    Args:
        note_id: ID of note
    
    Returns:
        List of label IDs (empty list if no labels)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT LabelID
            FROM dbo.APP_DrawerNoteLabelLink
            WHERE NoteID = ?
            ORDER BY LabelID
        """
        
        cursor.execute(query, (note_id,))
        rows = cursor.fetchall()
        
        return [row.LabelID for row in rows]
        
    finally:
        cursor.close()
        conn.close()


def filter_notes_by_label_ids(
    label_ids: List[int],
    limit: int,
    offset: int,
    patient_admission_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Filter non-deleted notes that have ALL specified labels (AND logic).
    Uses GROUP BY + HAVING COUNT to enforce AND logic.
    
    Args:
        label_ids: List of label IDs (note must have ALL of these)
        limit: Maximum number of notes to return
        offset: Number of notes to skip
        patient_admission_id: Optional filter by patient
    
    Returns:
        List of note dicts matching all labels
    """
    if not label_ids:
        # No labels specified - return empty list
        return []
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build parameterized query with correct number of placeholders
        label_count = len(label_ids)
        placeholders = ','.join(['?'] * label_count)
        
        patient_filter = ""
        params = list(label_ids)
        
        if patient_admission_id is not None:
            patient_filter = "AND n.PatientAdmissionID = ?"
            params.append(patient_admission_id)
        
        query = f"""
            SELECT 
                n.NoteID,
                n.NoteText,
                n.CreatedAt,
                n.CreatedByUserID,
                n.CreatedByName,
                n.IsDeleted,
                n.PatientAdmissionID,
                COALESCE(p.FullName, r.FullName) as PatientName
            FROM dbo.APP_DrawerNote n
            LEFT JOIN dbo.APP_VIEWTABLE_PATIENT_ADMISSION p ON n.PatientAdmissionID = p.PatientAdmissionID
            LEFT JOIN dbo.APP_RESERVE_PATIENT r ON n.PatientAdmissionID = r.PatientAdmissionID AND p.PatientAdmissionID IS NULL
            INNER JOIN dbo.APP_DrawerNoteLabelLink lnk ON n.NoteID = lnk.NoteID
            WHERE n.IsDeleted = 0
            AND lnk.LabelID IN ({placeholders})
            {patient_filter}
            GROUP BY n.NoteID, n.NoteText, n.CreatedAt, n.CreatedByUserID, n.CreatedByName, n.IsDeleted, n.PatientAdmissionID, COALESCE(p.FullName, r.FullName)
            HAVING COUNT(DISTINCT lnk.LabelID) = ?
            ORDER BY n.CreatedAt DESC
            OFFSET ? ROWS
            FETCH NEXT ? ROWS ONLY
        """
        
        # Parameters: label_ids + (patient_admission_id if provided) + label_count + offset + limit
        params.extend([label_count, offset, limit])
        
        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        
        notes = []
        for row in rows:
            notes.append({
                'note_id': row.NoteID,
                'note_text': row.NoteText,
                'created_at': row.CreatedAt,
                'created_by_user_id': row.CreatedByUserID,
                'created_by_name': row.CreatedByName,
                'is_deleted': row.IsDeleted,
                'patient_admission_id': row.PatientAdmissionID,
                'patient_name': row.PatientName
            })
        
        return notes
        
    finally:
        cursor.close()
        conn.close()


def get_all_notes_with_labels() -> List[Dict[str, Any]]:
    """
    Get all non-deleted notes with their label names for export.
    
    Returns:
        List of note dicts with label_names field:
        [
            {
                'note_id': int,
                'note_text': str,
                'created_at': datetime,
                'created_by_name': str,
                'label_names': list[str]  # Empty list if no labels
            }
        ]
        Ordered by created_at DESC
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get all non-deleted notes ordered by created_at DESC
        query = """
            SELECT 
                NoteID,
                NoteText,
                CreatedAt,
                CreatedByName
            FROM dbo.APP_DrawerNote
            WHERE IsDeleted = 0
            ORDER BY CreatedAt DESC
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        notes = []
        for row in rows:
            note_id = row.NoteID
            
            # Get label names for this note
            label_query = """
                SELECT lbl.LabelName
                FROM dbo.APP_DrawerNoteLabelLink lnk
                INNER JOIN dbo.APP_DrawerLabel lbl ON lnk.LabelID = lbl.LabelID
                WHERE lnk.NoteID = ?
                ORDER BY lbl.LabelName
            """
            
            cursor.execute(label_query, (note_id,))
            label_rows = cursor.fetchall()
            label_names = [lbl_row.LabelName for lbl_row in label_rows]
            
            notes.append({
                'note_id': note_id,
                'note_text': row.NoteText,
                'created_at': row.CreatedAt,
                'created_by_name': row.CreatedByName,
                'label_names': label_names
            })
        
        return notes
        
    finally:
        cursor.close()
        conn.close()
