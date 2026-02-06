"""
DB Layer for User Management Operations
Handles database operations for user creation, deletion and safety checks.

⚠️ ADMIN TEST TOOL — USER DELETE — HANDLE WITH CARE
"""

from typing import Optional, Any


def get_user_by_id(conn, user_id: int) -> Optional[Any]:
    """
    Get user information by UserID.
    
    Args:
        conn: Active database connection
        user_id: User's ID
        
    Returns:
        Row object with UserID and Username, or None if not found
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT UserID, Username
            FROM dbo.APP_Users
            WHERE UserID = ?
        """
        
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        
        return row
        
    finally:
        cursor.close()


def user_has_software_admin_role(conn, user_id: int) -> bool:
    """
    Check if user has SOFTWARE_ADMIN role assigned.
    
    Args:
        conn: Active database connection
        user_id: User's ID
        
    Returns:
        True if user has SOFTWARE_ADMIN role, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT COUNT(*) AS role_count
            FROM dbo.APP_UserRoleScope urs
            INNER JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE urs.UserID = ?
              AND r.RoleCode = 'SOFTWARE_ADMIN'
        """
        
        cursor.execute(query, (user_id,))
        result = cursor.fetchone()
        
        return result.role_count > 0
        
    finally:
        cursor.close()


def delete_user_scopes(conn, user_id: int) -> None:
    """
    Delete all role scope assignments for a user.
    
    Args:
        conn: Active database connection
        user_id: User's ID
        
    Note:
        Does NOT commit - caller controls transaction
        Must be called BEFORE delete_user
    """
    cursor = conn.cursor()
    
    try:
        query = """
            DELETE FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
        """
        
        cursor.execute(query, (user_id,))
        
    finally:
        cursor.close()


def delete_user(conn, user_id: int) -> None:
    """
    Delete user record from APP_Users table.
    
    Args:
        conn: Active database connection
        user_id: User's ID
        
    Note:
        Does NOT commit - caller controls transaction
        Must be called AFTER delete_user_scopes
    """
    cursor = conn.cursor()
    
    try:
        query = """
            DELETE FROM dbo.APP_Users
            WHERE UserID = ?
        """
        
        cursor.execute(query, (user_id,))
        
    finally:
        cursor.close()


def insert_user_record(
    conn,
    *,
    username: str,
    password_hash: str,
    display_name: Optional[str],
    department_display_name: Optional[str],
) -> int:
    """
    Insert standalone user record. No role/scope assignment.
    
    Args:
        conn: Active database connection
        username: Username for the new user (will be trimmed)
        password_hash: Password hash (already hashed by caller)
        display_name: Optional display name for UI greetings
        department_display_name: Optional department display label
        
    Returns:
        int: Newly created UserID
        
    Raises:
        ValueError: If username is empty after trimming
        
    Note:
        Does NOT commit - caller controls transaction
        Does NOT hash password - expects pre-hashed value
        Does NOT assign roles or scopes
    """
    # Defensive check - trim and validate username
    username = username.strip()
    if not username:
        raise ValueError("Username cannot be empty")
    
    cursor = conn.cursor()
    
    try:
        # Insert user with OUTPUT clause to get new ID
        query = """
            INSERT INTO dbo.APP_Users (
                Username,
                PasswordHash,
                DisplayName,
                DepartmentDisplayName,
                IsActive,
                CreatedAt
            )
            OUTPUT INSERTED.UserID
            VALUES (?, ?, ?, ?, 1, GETDATE())
        """
        
        cursor.execute(query, (
            username,
            password_hash,
            display_name,
            department_display_name
        ))
        
        result = cursor.fetchone()
        
        if not result or not result[0]:
            raise Exception(f"Failed to insert user '{username}' - no UserID returned")
        
        user_id = int(result[0])
        
        return user_id
        
    finally:
        cursor.close()


def insert_user_role_scope(
    conn,
    *,
    user_id: int,
    role_id: int,
    org_unit_id: int,
) -> None:
    """
    Assign role+scope to user if not already assigned.
    
    Args:
        conn: Active database connection
        user_id: UserID from APP_Users
        role_id: RoleID from APP_Roles
        org_unit_id: Organization unit ID (from AdminsrationUnit.UniqueID)
        
    Raises:
        ValueError: If any ID is <= 0
        
    Note:
        Does NOT commit - caller controls transaction
        Prevents duplicate assignments (idempotent)
        Does NOT validate that IDs exist in their respective tables
    """
    # Defensive checks
    if user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id} (must be > 0)")
    if role_id <= 0:
        raise ValueError(f"Invalid role_id: {role_id} (must be > 0)")
    if org_unit_id <= 0:
        raise ValueError(f"Invalid org_unit_id: {org_unit_id} (must be > 0)")
    
    cursor = conn.cursor()
    
    try:
        # Check if assignment already exists
        check_query = """
            SELECT 1
            FROM dbo.APP_UserRoleScope
            WHERE UserID = ?
              AND RoleID = ?
              AND OrgUnitID = ?
        """
        
        cursor.execute(check_query, (user_id, role_id, org_unit_id))
        existing = cursor.fetchone()
        
        # If already exists, do nothing (idempotent)
        if existing:
            return
        
        # Insert new role scope assignment
        insert_query = """
            INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType)
            VALUES (?, ?, ?, 'GENERIC')
        """
        
        cursor.execute(insert_query, (user_id, role_id, org_unit_id))
        
    finally:
        cursor.close()


def update_user_identity_fields(
    conn,
    *,
    user_id: int,
    display_name: Optional[str],
    department_display_name: Optional[str],
) -> None:
    """
    Update display identity fields for user. Identity only.
    
    Args:
        conn: Active database connection
        user_id: UserID to update
        display_name: New display name (None = no change)
        department_display_name: New department display name (None = no change)
        
    Raises:
        ValueError: If user_id <= 0 or user not found
        
    Note:
        Does NOT commit - caller controls transaction
        Uses COALESCE pattern for partial updates
        Only updates DisplayName and DepartmentDisplayName
        Does NOT update Username, PasswordHash, roles, or scopes
    """
    # Defensive check
    if user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id} (must be > 0)")
    
    cursor = conn.cursor()
    
    try:
        # Update using COALESCE pattern for partial updates
        # COALESCE(?, column) means: use new value if provided, else keep current value
        update_query = """
            UPDATE dbo.APP_Users
            SET DisplayName = COALESCE(?, DisplayName),
                DepartmentDisplayName = COALESCE(?, DepartmentDisplayName)
            WHERE UserID = ?
        """
        
        cursor.execute(update_query, (display_name, department_display_name, user_id))
        
        # Check if user was found and updated
        if cursor.rowcount == 0:
            raise ValueError(f"User with UserID {user_id} not found")
        
    finally:
        cursor.close()
