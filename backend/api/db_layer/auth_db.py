"""
Authentication Database Layer
Handles user authentication and role loading from SQL Server.

This module provides pure data access for authentication operations.
No business logic, no FastAPI dependencies - only database queries.

Tables used:
- APP_Users: System users with credentials
- APP_Roles: Role definitions
- APP_UserRoleScope: User-role-organizational unit mappings
"""

import pyodbc
import bcrypt
from typing import Optional, Dict, Any, List


def get_connection():
    """Get SQL Server database connection."""
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


# ==================== USER LOADING ====================

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Load a user by UserID without role scopes.
    
    Args:
        user_id: User ID to look up
        
    Returns:
        User dict with basic info, or None if not found
        
    Example return:
        {
            "user_id": 5,
            "username": "section_admin",
            "is_active": True,
            "scopes": []
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                UserID,
                Username,
                IsActive
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            "user_id": row.UserID,
            "username": row.Username,
            "is_active": bool(row.IsActive),
            "scopes": []  # Basic load doesn't include scopes
        }
        
    except Exception as e:
        raise Exception(f"Failed to load user by ID {user_id}: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Load a user by Username without role scopes.
    
    Args:
        username: Username to look up
        
    Returns:
        User dict with basic info, or None if not found
        
    Example return:
        {
            "user_id": 5,
            "username": "section_admin",
            "is_active": True,
            "scopes": []
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                UserID,
                Username,
                IsActive
            FROM dbo.APP_Users
            WHERE Username = ?
        """, (username,))
        
        row = cursor.fetchone()
        
        if not row:
            return None
        
        return {
            "user_id": row.UserID,
            "username": row.Username,
            "is_active": bool(row.IsActive),
            "scopes": []  # Basic load doesn't include scopes
        }
        
    except Exception as e:
        raise Exception(f"Failed to load user by username '{username}': {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_user_with_scopes(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Load a user by UserID WITH all role scopes.
    
    This is the main function for loading authenticated users.
    Uses a single JOIN query to fetch user + roles + scopes efficiently.
    
    Args:
        user_id: User ID to look up
        
    Returns:
        User dict with roles and scopes, or None if user not found
        
    Example return:
        {
            "user_id": 5,
            "username": "section_admin",
            "display_name": "Dr. Sarah Johnson",
            "department_display_name": "Emergency Medicine",
            "is_active": True,
            "scopes": [
                {
                    "role_code": "SECTION_ADMIN",
                    "org_unit_id": 10,
                    "org_unit_type": "SECTION"
                },
                {
                    "role_code": "WORKER",
                    "org_unit_id": 10,
                    "org_unit_type": "SECTION"
                }
            ]
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # First, check if user exists and is active
        cursor.execute("""
            SELECT 
                UserID,
                Username,
                DisplayName,
                DepartmentDisplayName,
                IsActive
            FROM dbo.APP_Users
            WHERE UserID = ?
        """, (user_id,))
        
        user_row = cursor.fetchone()
        
        if not user_row:
            return None
        
        user_data = {
            "user_id": user_row.UserID,
            "username": user_row.Username,
            "display_name": user_row.DisplayName,
            "department_display_name": user_row.DepartmentDisplayName,
            "is_active": bool(user_row.IsActive),
            "scopes": []
        }
        
        # Load all role scopes for this user
        cursor.execute("""
            SELECT 
                r.RoleCode,
                urs.OrgUnitID,
                urs.OrgUnitType
            FROM dbo.APP_UserRoleScope urs
            INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE urs.UserID = ?
            ORDER BY r.RoleCode, urs.OrgUnitType, urs.OrgUnitID
        """, (user_id,))
        
        scope_rows = cursor.fetchall()
        
        for scope_row in scope_rows:
            user_data["scopes"].append({
                "role_code": scope_row.RoleCode,
                "org_unit_id": scope_row.OrgUnitID,
                "org_unit_type": scope_row.OrgUnitType
            })
        
        return user_data
        
    except Exception as e:
        raise Exception(f"Failed to load user with scopes for ID {user_id}: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== AUTHENTICATION ====================

def validate_user_credentials(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Validate user credentials and return user with scopes if valid.
    
    This function:
    1. Loads user by username
    2. Checks if user is active
    3. Verifies password using bcrypt
    4. If valid, loads and returns user with all role scopes
    
    Args:
        username: Username to authenticate
        password: Plain text password to verify
        
    Returns:
        User dict with roles and scopes if credentials valid, None otherwise
        
    Example return on success:
        {
            "user_id": 5,
            "username": "section_admin",
            "is_active": True,
            "scopes": [
                {
                    "role_code": "SECTION_ADMIN",
                    "org_unit_id": 10,
                    "org_unit_type": "SECTION"
                }
            ]
        }
        
    Example return on failure:
        None
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Load user with password hash
        cursor.execute("""
            SELECT 
                UserID,
                Username,
                PasswordHash,
                IsActive
            FROM dbo.APP_Users
            WHERE Username = ?
        """, (username,))
        
        row = cursor.fetchone()
        
        # User not found
        if not row:
            return None
        
        user_id = row.UserID
        stored_username = row.Username
        password_hash = row.PasswordHash
        is_active = bool(row.IsActive)
        
        # User exists but is inactive
        if not is_active:
            return None
        
        # Check if password hash is a temporary placeholder
        # (from initial seed data - needs to be replaced with real hash)
        if password_hash.startswith('TEMP_HASH_'):
            # For testing: extract expected password from placeholder
            # Format: TEMP_HASH_<password>
            expected_password = password_hash.replace('TEMP_HASH_', '')
            
            # Simple string comparison for temp hashes
            if password != expected_password:
                return None
        else:
            # Verify actual bcrypt hash
            try:
                # Convert password to bytes if needed
                password_bytes = password.encode('utf-8') if isinstance(password, str) else password
                hash_bytes = password_hash.encode('utf-8') if isinstance(password_hash, str) else password_hash
                
                if not bcrypt.checkpw(password_bytes, hash_bytes):
                    return None
            except Exception as e:
                # bcrypt verification failed
                raise Exception(f"Password verification error: {str(e)}")
        
        # Password valid! Now load user with all scopes
        # Close current cursor and connection, use get_user_with_scopes
        cursor.close()
        conn.close()
        cursor = None
        conn = None
        
        # Load full user with scopes
        user_with_scopes = get_user_with_scopes(user_id)
        
        return user_with_scopes
        
    except Exception as e:
        raise Exception(f"Failed to validate credentials for user '{username}': {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


# ==================== UTILITY FUNCTIONS ====================

def hash_password(password: str) -> str:
    """
    Hash a plain text password using bcrypt.
    
    This is a utility function for creating password hashes.
    Typically used for:
    - Initial user creation
    - Password reset operations
    - Replacing temporary seed data hashes
    
    Args:
        password: Plain text password
        
    Returns:
        Bcrypt hash as string
        
    Example:
        >>> hash_password("admin123")
        '$2b$12$KIXxFz...'
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')


def update_user_password(user_id: int, new_password_hash: str) -> bool:
    """
    Update a user's password hash in the database.
    
    This is a utility function for password management.
    The hash should be generated using hash_password() or bcrypt directly.
    
    Args:
        user_id: User ID to update
        new_password_hash: New bcrypt hash to store
        
    Returns:
        True if update successful, False if user not found
        
    Raises:
        Exception: If database operation fails
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE dbo.APP_Users
            SET PasswordHash = ?
            WHERE UserID = ?
        """, (new_password_hash, user_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to update password for user ID {user_id}: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def update_user_profile(
    user_id: int,
    display_name: Optional[str] = None,
    department_display_name: Optional[str] = None
) -> bool:
    """
    Update a user's profile information (display fields).
    
    Phase A: Supports updating DisplayName and DepartmentDisplayName.
    Uses COALESCE to preserve existing values when parameter is None.
    
    Args:
        user_id: User ID to update
        display_name: New display name (None = keep existing value)
        department_display_name: New department display name (None = keep existing value)
        
    Returns:
        True if update successful, False if user not found
        
    Raises:
        Exception: If database operation fails
        
    Example:
        >>> # Update only display name
        >>> update_user_profile(5, display_name="Ahmed Al-Rashid")
        
        >>> # Update both fields
        >>> update_user_profile(5, "Ahmed", "Emergency Dept")
        
        >>> # Update only department
        >>> update_user_profile(5, department_display_name="Cardiology")
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Use COALESCE to keep existing values when parameter is None
        cursor.execute("""
            UPDATE dbo.APP_Users
            SET DisplayName = COALESCE(?, DisplayName),
                DepartmentDisplayName = COALESCE(?, DepartmentDisplayName)
            WHERE UserID = ?
        """, (display_name, department_display_name, user_id))
        
        rows_affected = cursor.rowcount
        conn.commit()
        
        return rows_affected > 0
        
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Failed to update profile for user ID {user_id}: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
