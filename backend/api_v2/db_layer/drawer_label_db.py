"""
Drawer Label Database Layer (API V2 - Phase G)
Handles SQL operations for APP_DrawerLabel table.

This is part of Phase G Drawer Notes multi-label system.
NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Dict, Any, List, Set
from core.database import get_connection


# ============================================================
# LABEL CRUD OPERATIONS
# ============================================================

def insert_label(label_name: str) -> int:
    """
    Create a new drawer label.
    
    Args:
        label_name: Label name (must be unique)
    
    Returns:
        LabelID of newly created label
    
    Raises:
        Exception: If insert fails (e.g., duplicate label name)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            INSERT INTO dbo.APP_DrawerLabel (LabelName)
            OUTPUT INSERTED.LabelID
            VALUES (?)
        """
        
        cursor.execute(query, (label_name,))
        label_id = cursor.fetchone()[0]
        conn.commit()
        
        return label_id
        
    finally:
        cursor.close()
        conn.close()


def list_active_labels() -> List[Dict[str, Any]]:
    """
    Get all active labels, ordered by label name.
    
    Returns:
        List of label dicts (only active labels where IsActive = 1)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                LabelID,
                LabelName,
                IsActive,
                CreatedAt
            FROM dbo.APP_DrawerLabel
            WHERE IsActive = 1
            ORDER BY LabelName
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        labels = []
        for row in rows:
            labels.append({
                'label_id': row.LabelID,
                'label_name': row.LabelName,
                'is_active': row.IsActive,
                'created_at': row.CreatedAt
            })
        
        return labels
        
    finally:
        cursor.close()
        conn.close()


def disable_label(label_id: int) -> None:
    """
    Disable a label by setting IsActive = 0.
    Disabled labels won't appear in active label lists.
    
    Args:
        label_id: ID of label to disable
    
    Raises:
        Exception: If update fails
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        query = """
            UPDATE dbo.APP_DrawerLabel
            SET IsActive = 0
            WHERE LabelID = ?
        """
        
        cursor.execute(query, (label_id,))
        conn.commit()
        
    finally:
        cursor.close()
        conn.close()


def get_label_ids_exist(label_ids: List[int]) -> Set[int]:
    """
    Check which label IDs exist and are active.
    Used for validation before attaching labels to notes.
    
    Args:
        label_ids: List of label IDs to check
    
    Returns:
        Set of label IDs that exist and are active
    """
    if not label_ids:
        return set()
    
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Build parameterized query with correct number of placeholders
        placeholders = ','.join(['?'] * len(label_ids))
        
        query = f"""
            SELECT LabelID
            FROM dbo.APP_DrawerLabel
            WHERE LabelID IN ({placeholders})
            AND IsActive = 1
        """
        
        cursor.execute(query, tuple(label_ids))
        rows = cursor.fetchall()
        
        return {row.LabelID for row in rows}
        
    finally:
        cursor.close()
        conn.close()
