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
    email: Optional[str] = None,
) -> int:
    """
    Insert standalone user record. No role/scope assignment.
    
    Args:
        conn: Active database connection
        username: Username for the new user (will be trimmed)
        password_hash: Password hash (already hashed by caller)
        display_name: Optional display name for UI greetings
        department_display_name: Optional department display label
        email: Optional email address for notifications
        
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
                Email,
                IsActive,
                CreatedAt
            )
            OUTPUT INSERTED.UserID
            VALUES (?, ?, ?, ?, ?, 1, GETDATE())
        """
        
        cursor.execute(query, (
            username,
            password_hash,
            display_name,
            department_display_name,
            email
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
    email: Optional[str] = None,
) -> None:
    """
    Update display identity fields for user. Identity only.
    
    Args:
        conn: Active database connection
        user_id: UserID to update
        display_name: New display name (None = no change)
        department_display_name: New department display name (None = no change)
        email: Email address for notifications (None = no change)
        
    Raises:
        ValueError: If user_id <= 0 or user not found
        
    Note:
        Does NOT commit - caller controls transaction
        Uses COALESCE pattern for partial updates
        Only updates DisplayName, DepartmentDisplayName, Email
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
                DepartmentDisplayName = COALESCE(?, DepartmentDisplayName),
                Email = COALESCE(?, Email)
            WHERE UserID = ?
        """
        
        cursor.execute(update_query, (display_name, department_display_name, email, user_id))
        
        # Check if user was found and updated
        if cursor.rowcount == 0:
            raise ValueError(f"User with UserID {user_id} not found")
        
    finally:
        cursor.close()


def get_user_with_role(conn, user_id: int) -> Optional[Any]:
    """
    Get user information with their role for edit operations.
    
    Args:
        conn: Active database connection
        user_id: User's ID
        
    Returns:
        Row object with UserID, Username, DisplayName, and RoleCode, or None if not found
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT TOP 1
                u.UserID,
                u.Username,
                u.DisplayName,
                r.RoleCode
            FROM dbo.APP_Users u
            LEFT JOIN dbo.APP_UserRoleScope urs ON u.UserID = urs.UserID
            LEFT JOIN dbo.APP_Roles r ON urs.RoleID = r.RoleID
            WHERE u.UserID = ?
        """
        
        cursor.execute(query, (user_id,))
        row = cursor.fetchone()
        
        return row
        
    finally:
        cursor.close()


def username_exists_excluding_user(conn, username: str, exclude_user_id: int) -> bool:
    """
    Check if username exists for a different user.
    
    Args:
        conn: Active database connection
        username: Username to check
        exclude_user_id: UserID to exclude from check (current user being edited)
        
    Returns:
        True if username exists for a different user, False otherwise
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT COUNT(*) AS count
            FROM dbo.APP_Users
            WHERE Username = ?
              AND UserID != ?
        """
        
        cursor.execute(query, (username, exclude_user_id))
        result = cursor.fetchone()
        
        return result.count > 0
        
    finally:
        cursor.close()


def update_user_credentials(
    conn,
    *,
    user_id: int,
    username: Optional[str] = None,
    password_hash: Optional[str] = None,
    test_password: Optional[str] = None,
    display_name: Optional[str] = None,
) -> None:
    """
    Update user credentials (username, password, display_name).
    
    Args:
        conn: Active database connection
        user_id: UserID to update
        username: New username (None = no change)
        password_hash: New password hash (None = no change)
        test_password: Plain text password for test_password field (None = no change)
        display_name: New display name (None = no change)
        
    Raises:
        ValueError: If user_id <= 0 or user not found
        
    Note:
        Does NOT commit - caller controls transaction
        Updates only provided fields (partial updates)
        For test_password: stores TEMP_HASH_ + plain password for testing
    """
    # Defensive check
    if user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id} (must be > 0)")
    
    # Build dynamic UPDATE query based on what's provided
    set_clauses = []
    params = []
    
    if display_name is not None:
        set_clauses.append("DisplayName = ?")
        params.append(display_name)
    
    if username is not None:
        set_clauses.append("Username = ?")
        params.append(username)
    
    if password_hash is not None:
        set_clauses.append("PasswordHash = ?")
        # Store as TEMP_HASH_ format if test_password is provided, otherwise use the hash as-is
        if test_password is not None:
            params.append(f"TEMP_HASH_{test_password}")
        else:
            params.append(password_hash)
    
    # If nothing to update, return
    if not set_clauses:
        return
    
    params.append(user_id)
    
    cursor = conn.cursor()
    
    try:
        update_query = f"""
            UPDATE dbo.APP_Users
            SET {', '.join(set_clauses)}
            WHERE UserID = ?
        """
        
        cursor.execute(update_query, params)
        
        # Check if user was found and updated
        if cursor.rowcount == 0:
            raise ValueError(f"User with UserID {user_id} not found")
        
    finally:
        cursor.close()
