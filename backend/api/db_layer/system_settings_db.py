"""
System Settings Database Layer
Handles CRUD operations for APP_SystemSettings table.
"""

import pyodbc
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_connection():
    """Get database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


# =============================================
# SYSTEM SETTINGS - READ OPERATIONS
# =============================================

def get_all_system_settings() -> List[Dict[str, Any]]:
    """
    Fetch all system settings from the database.
    
    Returns:
        List of dictionaries with setting details
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        SettingKey,
        SettingValue,
        SettingType,
        Description,
        UpdatedAt,
        UpdatedByUserID
    FROM APP_SystemSettings
    ORDER BY SettingKey
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    settings = []
    for row in rows:
        settings.append({
            'key': row.SettingKey,
            'value': row.SettingValue,
            'type': row.SettingType,
            'description': row.Description,
            'updated_at': row.UpdatedAt.isoformat() if row.UpdatedAt else None,
            'updated_by_user_id': row.UpdatedByUserID
        })
    
    cursor.close()
    conn.close()
    
    return settings


def get_system_setting(key: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single system setting by key.
    
    Args:
        key: The setting key to retrieve
        
    Returns:
        Dictionary with setting details, or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        SettingKey,
        SettingValue,
        SettingType,
        Description,
        UpdatedAt,
        UpdatedByUserID
    FROM APP_SystemSettings
    WHERE SettingKey = ?
    """
    
    cursor.execute(query, (key,))
    row = cursor.fetchone()
    
    if not row:
        cursor.close()
        conn.close()
        return None
    
    setting = {
        'key': row.SettingKey,
        'value': row.SettingValue,
        'type': row.SettingType,
        'description': row.Description,
        'updated_at': row.UpdatedAt.isoformat() if row.UpdatedAt else None,
        'updated_by_user_id': row.UpdatedByUserID
    }
    
    cursor.close()
    conn.close()
    
    return setting


# =============================================
# SYSTEM SETTINGS - WRITE OPERATIONS
# =============================================

def update_system_setting(
    key: str,
    value: str,
    updated_by_user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Update a system setting value.
    
    Args:
        key: The setting key to update
        value: The new value (as string)
        updated_by_user_id: ID of the user making the update (optional)
        
    Returns:
        Dictionary with updated setting details
        
    Raises:
        ValueError: If the setting key doesn't exist
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # First check if the setting exists
    check_query = "SELECT SettingKey FROM APP_SystemSettings WHERE SettingKey = ?"
    cursor.execute(check_query, (key,))
    if not cursor.fetchone():
        cursor.close()
        conn.close()
        raise ValueError(f"Setting key '{key}' not found")
    
    # Update the setting
    update_query = """
    UPDATE APP_SystemSettings
    SET 
        SettingValue = ?,
        UpdatedAt = GETDATE(),
        UpdatedByUserID = ?
    WHERE SettingKey = ?
    """
    
    cursor.execute(update_query, (value, updated_by_user_id, key))
    conn.commit()
    
    cursor.close()
    conn.close()
    
    # Return the updated setting
    return get_system_setting(key)


def create_system_setting(
    key: str,
    value: str,
    setting_type: str,
    description: Optional[str] = None,
    updated_by_user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a new system setting.
    
    Args:
        key: The setting key (must be unique)
        value: The setting value
        setting_type: The type ('int', 'bool', 'string', 'json')
        description: Optional description
        updated_by_user_id: ID of the user creating the setting
        
    Returns:
        Dictionary with the new setting details
        
    Raises:
        ValueError: If the setting key already exists
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if the setting already exists
    check_query = "SELECT SettingKey FROM APP_SystemSettings WHERE SettingKey = ?"
    cursor.execute(check_query, (key,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        raise ValueError(f"Setting key '{key}' already exists")
    
    # Insert the new setting
    insert_query = """
    INSERT INTO APP_SystemSettings (
        SettingKey,
        SettingValue,
        SettingType,
        Description,
        UpdatedAt,
        UpdatedByUserID
    )
    VALUES (?, ?, ?, ?, GETDATE(), ?)
    """
    
    cursor.execute(insert_query, (key, value, setting_type, description, updated_by_user_id))
    conn.commit()
    
    cursor.close()
    conn.close()
    
    # Return the new setting
    return get_system_setting(key)


def delete_system_setting(key: str) -> bool:
    """
    Delete a system setting.
    
    Args:
        key: The setting key to delete
        
    Returns:
        True if deleted, False if not found
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    query = "DELETE FROM APP_SystemSettings WHERE SettingKey = ?"
    cursor.execute(query, (key,))
    rows_affected = cursor.rowcount
    
    conn.commit()
    cursor.close()
    conn.close()
    
    return rows_affected > 0
