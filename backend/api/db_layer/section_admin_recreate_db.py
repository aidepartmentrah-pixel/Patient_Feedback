"""
DB Layer for Section Admin Recreation
Handles database operations for recreating section admin users.

⚠️ ADMIN TEST TOOL — RECREATE SECTION ADMIN USER
"""

from typing import Optional, Any


def get_section(conn, section_id: int) -> Optional[Any]:
    """
    Get section information by section ID.
    
    Args:
        conn: Active database connection
        section_id: Section's UniqueID
        
    Returns:
        Row object with UniqueID, Name, Type, or None if not found
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT UniqueID, Name, Type
            FROM dbo.AdminsrationUnit
            WHERE UniqueID = ?
        """
        
        cursor.execute(query, (section_id,))
        row = cursor.fetchone()
        
        return row
        
    finally:
        cursor.close()


def username_exists(conn, username: str) -> bool:
    """
    Check if a username already exists in APP_Users.
    
    Args:
        conn: Active database connection
        username: Username to check
        
    Returns:
        True if username exists, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 1 FROM dbo.APP_Users WHERE Username = ?
        """
        
        cursor.execute(query, (username,))
        result = cursor.fetchone()
        
        return result is not None
        
    finally:
        cursor.close()


def insert_user(conn, username: str, display_name: str = None, department_display_name: str = None) -> int:
    """
    Insert a new user into APP_Users table.
    
    Args:
        conn: Active database connection
        username: Username for new user
        display_name: Optional display name for person greeting. Defaults to username if None.
        department_display_name: Optional department display label. Defaults to NULL if None.
        
    Returns:
        int: New user's UserID
        
    Note:
        Does NOT commit - caller controls transaction
        Uses TEMP_HASH password for testing phase
        Phase A: Now supports DisplayName and DepartmentDisplayName
    """
    cursor = conn.cursor()
    
    try:
        # Fallback: If display_name is None, use username
        effective_display_name = display_name if display_name is not None else username
        
        insert_query = """
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, CreatedAt, DisplayName, DepartmentDisplayName)
            VALUES (?, 'TEMP_HASH_Hospital2026!', 1, GETDATE(), ?, ?)
        """
        
        cursor.execute(insert_query, (username, effective_display_name, department_display_name))
        
        # Get the newly created user ID using @@IDENTITY
        cursor.execute("SELECT @@IDENTITY AS user_id")
        result = cursor.fetchone()
        
        if not result or result.user_id is None:
            raise Exception(f"Failed to get user_id for newly created user '{username}'")
        
        user_id = int(result.user_id)
        
        return user_id
        
    finally:
        cursor.close()


def insert_user_scope(conn, user_id: int, role_code: str, org_unit_id: int) -> None:
    """
    Insert a new user role scope into APP_UserRoleScope table.
    
    Args:
        conn: Active database connection
        user_id: UserID from APP_Users
        role_code: Role code (e.g., 'SECTION_ADMIN')
        org_unit_id: Section UniqueID
        
    Raises:
        Exception: If role_code not found
        
    Note:
        Does NOT commit - caller controls transaction
        Resolves RoleID dynamically from role_code
        Sets OrgUnitType to 'SECTION' (Type 324)
    """
    cursor = conn.cursor()
    
    try:
        # Resolve RoleID from RoleCode
        role_query = "SELECT RoleID FROM dbo.APP_Roles WHERE RoleCode = ?"
        cursor.execute(role_query, (role_code,))
        role_result = cursor.fetchone()
        
        if not role_result:
            raise Exception(f"Role code '{role_code}' not found")
        
        role_id = role_result.RoleID
        
        # Insert user scope with OrgUnitType as string
        scope_query = """
            INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
            VALUES (?, ?, ?, 'SECTION')
        """
        
        cursor.execute(scope_query, (user_id, role_id, org_unit_id))
        
    finally:
        cursor.close()
