"""
Service Layer for Section Admin Recreation
Handles recreation of section admin users with unique username generation.

⚠️ ADMIN TEST TOOL — RECREATE SECTION ADMIN USER
"""

from typing import Dict, Any
from core.database import get_connection
from ..constants.org_unit_types import ORG_TYPE_SECTION
from ..db_layer.section_admin_recreate_db import (
    get_section,
    username_exists,
    insert_user,
    insert_user_scope
)


def recreate_section_admin_service(section_id: int) -> Dict[str, Any]:
    """
    Create a new section admin user for an existing section.
    
    Args:
        section_id: ID of section (AdminsrationUnit.UniqueID)
        
    Returns:
        dict: Contains section_id, username, and temp_password
        
    Raises:
        Exception: If section not found
        Exception: If section Type is not 324 (SECTION)
        Exception: If username generation fails
        Exception: If database operation fails (transaction rolled back)
        
    Process:
        1. Verify section exists and Type = 324
        2. Generate unique username with version suffix (sec_{id}_admin_v2, v3, etc.)
        3. Insert new user with TEMP_HASH password
        4. Assign SECTION_ADMIN role scope
        5. Commit transaction
        
    Note:
        Does NOT delete or modify existing section admins
        Creates additional admin account for the section
    """
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Step 1: Verify section exists
        section = get_section(conn, section_id)
        
        if not section:
            raise Exception(f"Section with ID {section_id} not found")
        
        section_type = section.Type
        section_name = section.Name
        
        # Verify Type = SECTION
        if section_type != ORG_TYPE_SECTION:
            raise Exception(
                f"Organization unit '{section_name}' (ID {section_id}) is not a section. "
                f"Type is {section_type}, expected {ORG_TYPE_SECTION} (SECTION)."
            )
        
        # Step 2: Generate unique username with version suffix
        base_username = f"sec_{section_id}_admin"
        
        # Check if base username is available
        if not username_exists(conn, base_username):
            new_username = base_username
        else:
            # Generate versioned username (v2, v3, v4, etc.)
            version = 2
            max_attempts = 100  # Safety limit
            
            while version <= max_attempts:
                candidate_username = f"{base_username}_v{version}"
                
                if not username_exists(conn, candidate_username):
                    new_username = candidate_username
                    break
                
                version += 1
            else:
                # Reached max attempts without finding unique username
                raise Exception(
                    f"Failed to generate unique username for section {section_id}. "
                    f"Tried {max_attempts} versions."
                )
        
        # Step 3: Create new user
        user_id = insert_user(conn, new_username)
        
        # Step 4: Assign SECTION_ADMIN role scope
        insert_user_scope(conn, user_id, "SECTION_ADMIN", section_id)
        
        # Step 5: Commit transaction
        conn.commit()
        
        # Return credentials (password without TEMP_HASH_ prefix)
        return {
            "section_id": section_id,
            "username": new_username,
            "temp_password": "Hospital2026!"
        }
        
    except Exception as e:
        # Rollback on error
        if conn:
            conn.rollback()
        raise Exception(f"Failed to recreate section admin: {str(e)}")
        
    finally:
        # Always close connection
        if conn:
            conn.close()
