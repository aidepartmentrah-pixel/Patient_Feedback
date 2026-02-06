"""
Service Layer for User Management Operations
Handles user creation, deletion with safety checks and transaction management.

⚠️ ADMIN TEST TOOL — USER DELETE — HANDLE WITH CARE
"""

from typing import Dict, Any, Optional
from core.database import get_connection
from ..db_layer.user_management_db import (
    get_user_by_id,
    user_has_software_admin_role,
    delete_user_scopes,
    delete_user,
    insert_user_record,
    insert_user_role_scope,
    update_user_identity_fields
)
from ..db_layer.section_admin_recreate_db import username_exists
from ..db_layer.auth_db import hash_password


def delete_user_service(user_id: int) -> Dict[str, Any]:
    """
    Delete a user and all their role scope assignments.
    
    Args:
        user_id: ID of user to delete
        
    Returns:
        dict: Contains deleted_user_id and deleted_username
        
    Raises:
        Exception: If user not found
        Exception: If user is protected (software_admin or has SOFTWARE_ADMIN role)
        Exception: If database operation fails (transaction rolled back)
        
    Safety Rules:
        - Blocks deletion of username "software_admin"
        - Blocks deletion of any user with SOFTWARE_ADMIN role
        - Deletes role scopes before deleting user
        - All operations run in transaction
        
    Process:
        1. Load user information
        2. Check protection rules
        3. Delete user scopes (APP_UserRoleScope)
        4. Delete user record (APP_Users)
        5. Commit transaction
    """
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Step 1: Load user
        user = get_user_by_id(conn, user_id)
        
        if not user:
            raise Exception(f"User with ID {user_id} not found")
        
        username = user.Username
        
        # Step 2: HARD BLOCK rules - protect critical accounts
        
        # Block software_admin username (primary system admin account)
        if username.lower() == "software_admin":
            raise Exception(
                f"Cannot delete protected account 'software_admin'. "
                f"This is the primary system administrator account."
            )
        
        # Block any user with SOFTWARE_ADMIN role
        if user_has_software_admin_role(conn, user_id):
            raise Exception(
                f"Cannot delete user '{username}' because they have SOFTWARE_ADMIN role. "
                f"Remove SOFTWARE_ADMIN role first or use a different method."
            )
        
        # Step 3: Delete user scopes first (foreign key constraint)
        delete_user_scopes(conn, user_id)
        
        # Step 4: Delete user record
        delete_user(conn, user_id)
        
        # Step 5: Commit transaction
        conn.commit()
        
        # Return deletion confirmation
        return {
            "deleted_user_id": user_id,
            "deleted_username": username
        }
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        raise Exception(f"Failed to delete user: {str(e)}")
        
    finally:
        # Always close connection
        if conn:
            conn.close()


def create_user_with_role_scope(
    *,
    username: str,
    password_plain: str,
    display_name: Optional[str],
    department_display_name: Optional[str],
    role_id: int,
    org_unit_id: int,
) -> int:
    """
    Create standalone user and assign single role+scope.
    
    Args:
        username: Username for new user (will be trimmed)
        password_plain: Plain text password (will be hashed with bcrypt)
        display_name: Optional display name for UI greetings
        department_display_name: Optional department display label
        role_id: RoleID from APP_Roles to assign
        org_unit_id: Organization unit ID to assign
        
    Returns:
        int: Newly created UserID
        
    Raises:
        ValueError: If inputs are invalid or username already exists
        Exception: If database operation fails (transaction rolled back)
        
    Process:
        1. Validate inputs
        2. Normalize username (trim)
        3. Check username uniqueness
        4. Hash password with bcrypt
        5. Insert user record
        6. Assign role+scope
        7. Commit transaction
        
    Note:
        - Creates user in transaction (commits at end)
        - Assigns exactly one role+scope combination
        - Does NOT send emails or create tokens
    """
    # Validate inputs
    if not username or not username.strip():
        raise ValueError("Username cannot be empty")
    
    if not password_plain or not password_plain.strip():
        raise ValueError("Password cannot be empty")
    
    if role_id <= 0:
        raise ValueError(f"Invalid role_id: {role_id} (must be > 0)")
    
    if org_unit_id <= 0:
        raise ValueError(f"Invalid org_unit_id: {org_unit_id} (must be > 0)")
    
    # Normalize username
    username = username.strip()
    
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Check username uniqueness
        if username_exists(conn, username):
            raise ValueError(f"Username '{username}' already exists")
        
        # Hash password
        password_hash = hash_password(password_plain)
        
        # Insert user record
        user_id = insert_user_record(
            conn,
            username=username,
            password_hash=password_hash,
            display_name=display_name,
            department_display_name=department_display_name
        )
        
        # Assign role+scope
        insert_user_role_scope(
            conn,
            user_id=user_id,
            role_id=role_id,
            org_unit_id=org_unit_id
        )
        
        # Commit transaction
        conn.commit()
        
        return user_id
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        # Re-raise exception
        raise
        
    finally:
        # Always close connection
        if conn:
            conn.close()


def update_user_identity_service(
    *,
    user_id: int,
    display_name: Optional[str],
    department_display_name: Optional[str],
) -> None:
    """
    Service: update user display identity fields only.
    
    Args:
        user_id: UserID to update
        display_name: New display name (None = no change)
        department_display_name: New department display name (None = no change)
        
    Raises:
        ValueError: If user_id <= 0, both fields are None, or user not found
        Exception: If database operation fails (transaction rolled back)
        
    Process:
        1. Validate inputs
        2. Normalize values (strip whitespace)
        3. Call DB layer update function
        4. Commit transaction
        
    Note:
        - Does NOT update password, roles, or scopes
        - At least one field must be provided (not None)
        - Updates in transaction (commits at end)
    """
    # Validate inputs
    if user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id} (must be > 0)")
    
    # At least one field must be provided
    if display_name is None and department_display_name is None:
        raise ValueError("At least one field must be provided (display_name or department_display_name)")
    
    # Normalize values - strip whitespace if provided
    if display_name is not None:
        display_name = display_name.strip()
    
    if department_display_name is not None:
        department_display_name = department_display_name.strip()
    
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Call DB layer update function
        update_user_identity_fields(
            conn,
            user_id=user_id,
            display_name=display_name,
            department_display_name=department_display_name
        )
        
        # Commit transaction
        conn.commit()
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        # Re-raise exception
        raise
        
    finally:
        # Always close connection
        if conn:
            conn.close()


def list_users_for_settings_service() -> list[dict]:
    """
    Adapter: reshape inventory data for Settings users table.
    
    Converts user inventory data into flat list suitable for Settings Users UI.
    Returns one row per user-role-scope entry.
    
    Returns:
        List of dictionaries with user information:
        [
            {
                "user_id": 123,
                "username": "user@hospital",
                "display_name": "Dr. Ahmed",
                "department_display_name": "Emergency",
                "role_name": "SECTION_ADMIN",
                "org_unit_name": "قسم الطوارئ",
                "is_active": True
            },
            ...
        ]
        
    Note:
        - Skips org units with null users
        - Read-only operation (no transaction needed)
        - Each user-role-scope combination returns as separate row
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query gets all user-role-scope combinations with full details
        # Only includes rows where user exists (INNER JOIN instead of LEFT JOIN)
        query = """
            SELECT 
                u.UserID as user_id,
                u.Username as username,
                u.DisplayName as display_name,
                u.DepartmentDisplayName as department_display_name,
                r.RoleCode as role_name,
                org.Name as org_unit_name,
                u.IsActive as is_active
            FROM dbo.APP_Users u
            INNER JOIN dbo.APP_UserRoleScope urs 
                ON u.UserID = urs.UserID
            INNER JOIN dbo.APP_Roles r 
                ON urs.RoleID = r.RoleID
            INNER JOIN dbo.AdminsrationUnit org 
                ON urs.OrgUnitID = org.UniqueID
            WHERE org.Frozen = 0  -- Only active org units
            ORDER BY 
                u.Username,
                r.RoleCode,
                org.Name
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert pyodbc rows to list of dictionaries
        results = []
        for row in rows:
            results.append({
                "user_id": row.user_id,
                "username": row.username,
                "display_name": row.display_name if row.display_name else None,
                "department_display_name": row.department_display_name if row.department_display_name else None,
                "role_name": row.role_name,
                "org_unit_name": row.org_unit_name,
                "is_active": bool(row.is_active) if row.is_active is not None else False
            })
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to retrieve users for settings: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def admin_reset_user_password_service(user_id: int, new_password: str) -> None:
    """
    Admin password reset for user accounts.
    
    SOFTWARE_ADMIN only operation - resets user password without requiring old password.
    
    Args:
        user_id: User ID to reset password for
        new_password: New plain text password (will be hashed)
        
    Raises:
        ValueError: If user_id <= 0 or new_password is empty
        ValueError: If user not found
        Exception: If database operation fails
        
    Process:
        1. Validate inputs
        2. Hash new password with bcrypt
        3. Update password in database
        4. Commit transaction
        
    Note:
        - Router must validate SOFTWARE_ADMIN role before calling
        - Password is hashed using bcrypt
        - No old password verification required (admin override)
    """
    # Validate inputs
    if user_id <= 0:
        raise ValueError(f"Invalid user_id: {user_id} (must be > 0)")
    
    if not new_password or not new_password.strip():
        raise ValueError("Password cannot be empty")
    
    conn = None
    cursor = None
    
    try:
        # Hash the new password
        password_hash = hash_password(new_password)
        
        # Open connection
        conn = get_connection()
        cursor = conn.cursor()
        
        # Update password
        cursor.execute("""
            UPDATE dbo.APP_Users
            SET PasswordHash = ?
            WHERE UserID = ?
        """, (password_hash, user_id))
        
        rows_affected = cursor.rowcount
        
        # Check if user was found
        if rows_affected == 0:
            raise ValueError(f"User with UserID {user_id} not found")
        
        # Commit transaction
        conn.commit()
        
    except ValueError:
        # Re-raise validation errors
        if conn:
            conn.rollback()
        raise
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        raise Exception(f"Failed to reset password for user {user_id}: {str(e)}")
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
