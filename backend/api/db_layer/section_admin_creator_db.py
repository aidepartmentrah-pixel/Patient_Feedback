"""
DB Layer for Section + Admin User Creation
Handles database operations for creating sections with admin users.
"""

from typing import Any


def insert_section(conn, name: str, parent_department_id: int) -> int:
    """
    Insert a new section into AdminsrationUnit table.
    
    Args:
        conn: Active database connection
        name: Section name
        parent_department_id: Parent department UniqueID
        
    Returns:
        int: New section's UniqueID
        
    Note:
        Does NOT commit - caller controls transaction
    """
    cursor = conn.cursor()
    
    try:
        # Insert section with Type = 324 and get the new ID using OUTPUT
        insert_query = """
            INSERT INTO dbo.AdminsrationUnit (Name, ParentID, Type, Frozen, CreateDate)
            OUTPUT INSERTED.UniqueID
            VALUES (?, ?, 324, 0, GETDATE())
        """
        
        cursor.execute(insert_query, (name, parent_department_id))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            raise Exception(
                f"Failed to insert section '{name}'. "
                f"Parent department ID {parent_department_id} may not exist or INSERT failed."
            )
        
        section_id = int(result[0])
        
        return section_id
        
    finally:
        cursor.close()


def insert_user(conn, username: str, display_name: str = None, department_display_name: str = None) -> int:
    """
    Insert a new user into APP_Users table.
    
    Args:
        conn: Active database connection
        username: Username (must be unique)
        display_name: Optional display name for person greeting. Defaults to username if None.
        department_display_name: Optional department display label. Defaults to NULL if None.
        
    Returns:
        int: New user's UserID
        
    Raises:
        Exception: If username already exists
        
    Note:
        Does NOT commit - caller controls transaction
        Uses TEMP_HASH password for testing phase
        Phase A: Now supports DisplayName and DepartmentDisplayName
    """
    cursor = conn.cursor()
    
    try:
        # Check if username already exists
        check_query = "SELECT COUNT(*) AS user_count FROM dbo.APP_Users WHERE Username = ?"
        cursor.execute(check_query, (username,))
        result = cursor.fetchone()
        
        if result.user_count > 0:
            raise Exception(f"Username '{username}' already exists")
        
        # Fallback: If display_name is None, use username
        effective_display_name = display_name if display_name is not None else username
        
        # Insert new user with TEMP_HASH password and display fields
        insert_query = """
            INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, CreatedAt, DisplayName, DepartmentDisplayName)
            OUTPUT INSERTED.UserID
            VALUES (?, 'TEMP_HASH_Hospital2026!', 1, GETDATE(), ?, ?)
        """
        
        cursor.execute(insert_query, (username, effective_display_name, department_display_name))
        result = cursor.fetchone()
        
        if not result or not result[0]:
            raise Exception(f"Failed to create user '{username}'")
        
        user_id = int(result[0])
        
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
        
        # Insert user scope with OrgUnitType as string (matches existing pattern)
        scope_query = """
            INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
            VALUES (?, ?, ?, 'SECTION')
        """
        
        cursor.execute(scope_query, (user_id, role_id, org_unit_id))
        
    finally:
        cursor.close()
