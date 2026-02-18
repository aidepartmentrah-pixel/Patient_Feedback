"""
Service Layer for Section + Admin User Creation
Orchestrates transactional creation of sections with admin users.

Phase C - B-C4: Split into core creation + admin attachment for modularity.
Phase C - B-C8: Added scope verification and logging.
"""

import logging
from typing import Dict, Any, Tuple, Optional
from core.database import get_connection
from ..db_layer.section_admin_creator_db import (
    insert_section,
    insert_user,
    insert_user_scope,
    verify_user_scope
)
from ..constants.org_unit_types import (
    ORG_TYPE_ADMINISTRATION,
    ORG_TYPE_DEPARTMENT,
    ORG_TYPE_SECTION
)

# Initialize logger for section creation operations
logger = logging.getLogger("section_admin_creator")


def create_section_core(
    conn,
    section_name: str,
    parent_unit_id: int,
    created_by_user_id: Optional[int] = None
) -> int:
    """
    Core function: Create section org unit only (no user creation).
    
    Phase C - B-C4: Extracted from create_section_with_admin for modularity.
    This function can be used independently when section creation is needed
    without automatic admin user creation.
    
    Args:
        conn: Active database connection (transaction controlled by caller)
        section_name: Name of the new section
        parent_unit_id: Parent org unit ID (administration or department)
        created_by_user_id: Optional user ID for audit trail (future use)
        
    Returns:
        int: New section's UniqueID
        
    Raises:
        Exception: If parent doesn't exist or insert fails
        
    Note:
        Does NOT commit - caller controls transaction
        Does NOT create admin user - use attach_section_admin_user for that
    """
    # Validate parent exists and has correct type
    # Phase C - B-C9: Parent must be ADMINISTRATION (322) or DEPARTMENT (323)
    cursor = conn.cursor()
    try:
        check_query = "SELECT UniqueID, Type FROM dbo.AdminsrationUnit WHERE UniqueID = ?"
        cursor.execute(check_query, (parent_unit_id,))
        result = cursor.fetchone()
        
        if not result:
            raise Exception(f"Parent org unit with ID {parent_unit_id} not found")
        
        parent_type = result.Type
        
        # Validate parent type (must be ADMINISTRATION or DEPARTMENT, not SECTION)
        if parent_type not in [ORG_TYPE_ADMINISTRATION, ORG_TYPE_DEPARTMENT]:
            raise Exception(
                f"Invalid parent type. Parent org unit {parent_unit_id} has Type={parent_type}. "
                f"Sections can only be created under ADMINISTRATION (Type={ORG_TYPE_ADMINISTRATION}) "
                f"or DEPARTMENT (Type={ORG_TYPE_DEPARTMENT}), not SECTION (Type={ORG_TYPE_SECTION})."
            )
    finally:
        cursor.close()
    
    # Insert section using DB layer
    section_id = insert_section(conn, section_name, parent_unit_id)
    
    return section_id


def attach_section_admin_user(
    conn,
    section_id: int,
    created_by_user_id: Optional[int] = None
) -> Tuple[str, str]:
    """
    Attach admin user: Create SECTION_ADMIN user and assign scope.
    
    Phase C - B-C4: Extracted from create_section_with_admin for modularity.
    This function can be used independently to create additional admin users
    for existing sections.
    
    Args:
        conn: Active database connection (transaction controlled by caller)
        section_id: Existing section's UniqueID
        created_by_user_id: Optional user ID for audit trail (future use)
        
    Returns:
        tuple: (username, temp_password)
        
    Raises:
        Exception: If section doesn't exist or user creation fails
        
    Note:
        Does NOT commit - caller controls transaction
        Generates username based on section_id: sec_{id}_admin
    """
    # Validate section exists
    cursor = conn.cursor()
    try:
        check_query = "SELECT UniqueID FROM dbo.AdminsrationUnit WHERE UniqueID = ?"
        cursor.execute(check_query, (section_id,))
        if not cursor.fetchone():
            raise Exception(f"Section with ID {section_id} not found")
    finally:
        cursor.close()
    
    # Generate username based on section_id
    username = f"sec_{section_id}_admin"
    
    # Create user using DB layer
    user_id = insert_user(conn, username)
    
    # Assign SECTION_ADMIN role with section scope
    # Phase C - B-C8: Capture role_id for verification
    role_id = insert_user_scope(conn, user_id, "SECTION_ADMIN", section_id)
    
    # Phase C - B-C8: Verify scope assignment
    # This ensures data integrity before committing the transaction
    scope_verified = verify_user_scope(
        conn=conn,
        user_id=user_id,
        expected_role_id=role_id,
        expected_org_unit_id=section_id,
        expected_org_unit_type="SECTION"
    )
    
    if not scope_verified:
        # Verification failed - raise exception to trigger rollback
        raise Exception(
            f"Scope verification failed for user {user_id} (username: {username}). "
            f"Expected: RoleID={role_id}, OrgUnitID={section_id}, OrgUnitType='SECTION'. "
            f"Scope row not found or does not match expectations."
        )
    
    # Phase C - B-C8: Log successful scope assignment (debug level)
    logger.debug(
        "Section admin scope assigned",
        extra={
            "user_id": user_id,
            "username": username,
            "section_id": section_id,
            "role_id": role_id,
            "org_unit_type": "SECTION"
        }
    )
    
    # Return credentials
    temp_password = "Hospital2026!"
    return (username, temp_password)


def create_section_with_admin(
    section_name: str,
    parent_department_id: int,
    create_admin: bool = True
) -> Dict[str, Any]:
    """
    Create a new section and optionally create a SECTION_ADMIN user for it.
    
    Phase C - B-C4: Orchestrates create_section_core + attach_section_admin_user.
    Phase C - B-C5: Added optional create_admin flag for internal flexibility.
    
    This is the main public function used by routers.
    
    Args:
        section_name: Name of the new section
        parent_department_id: Parent department's UniqueID
        create_admin: Whether to auto-create admin user (default: True)
        
    Returns:
        dict: Contains section_id, section_name, parent_unit_id, username, and temp_password
              If create_admin=False, username and temp_password will be None
        
    Raises:
        Exception: If any step fails (transaction rolled back)
        
    Process:
        1. Open transaction
        2. Create section org unit (create_section_core)
        3. Conditionally attach admin user if create_admin=True
        4. Commit transaction
        5. Return response (with or without credentials)
        
    Note:
        B-C5: Internal flexibility only - routers still pass create_admin=True by default.
        Contract maintained for backwards compatibility.
    """
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Step 1: Create section org unit (core function)
        section_id = create_section_core(
            conn=conn,
            section_name=section_name,
            parent_unit_id=parent_department_id,
            created_by_user_id=None  # Future: pass current user ID for audit
        )
        
        # Step 2: Conditionally attach admin user to section
        if create_admin:
            username, temp_password = attach_section_admin_user(
                conn=conn,
                section_id=section_id,
                created_by_user_id=None  # Future: pass current user ID for audit
            )
        else:
            # Section only - no user creation
            username = None
            temp_password = None
        
        # Step 3: Commit transaction
        conn.commit()
        
        # Return complete response with all created entity details
        return {
            "section_id": section_id,
            "section_name": section_name,
            "parent_unit_id": parent_department_id,
            "username": username,
            "temp_password": temp_password
        }
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        raise Exception(f"Failed to create section with admin: {str(e)}")
        
    finally:
        # Always close connection
        if conn:
            conn.close()
