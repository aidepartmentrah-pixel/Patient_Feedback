"""
Service Layer for Section + Admin User Creation
Orchestrates transactional creation of sections with admin users.
"""

from typing import Dict, Any
from core.database import get_connection
from ..db_layer.section_admin_creator_db import (
    insert_section,
    insert_user,
    insert_user_scope
)


def create_section_with_admin(section_name: str, parent_department_id: int) -> Dict[str, Any]:
    """
    Create a new section and automatically create a SECTION_ADMIN user for it.
    
    Args:
        section_name: Name of the new section
        parent_department_id: Parent department's UniqueID
        
    Returns:
        dict: Contains section_id, username, and temp_password
        
    Raises:
        Exception: If any step fails (transaction rolled back)
        
    Process:
        1. Insert section into AdminsrationUnit
        2. Generate username based on section_id
        3. Insert user into APP_Users
        4. Link user to section with SECTION_ADMIN role
        5. Commit transaction
    """
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Step 1: Create section
        section_id = insert_section(conn, section_name, parent_department_id)
        
        # Step 2: Generate username based on section_id
        username = f"sec_{section_id}_admin"
        
        # Step 3: Create user
        user_id = insert_user(conn, username)
        
        # Step 4: Assign SECTION_ADMIN role with section scope
        insert_user_scope(conn, user_id, "SECTION_ADMIN", section_id)
        
        # Step 5: Commit transaction
        conn.commit()
        
        # Return credentials (password without TEMP_HASH_ prefix)
        return {
            "section_id": section_id,
            "username": username,
            "temp_password": "Hospital2026!"
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
