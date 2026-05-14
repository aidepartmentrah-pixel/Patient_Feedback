"""
Service Layer for User Credentials Listing (TEST ONLY)
Handles password derivation from TEMP_HASH format.

⚠️ TEST MODE PASSWORD DERIVATION — DO NOT USE IN PRODUCTION
"""

from typing import List, Dict, Any, Optional
from core.database import get_connection
from ..db_layer.user_credentials_db import get_all_user_credentials


def get_all_user_credentials_service() -> List[Dict[str, Any]]:
    """
    Get all users with their credentials for testing purposes.
    
    Returns:
        List of user credential dictionaries with derived test passwords
        
    Note:
        TEST ONLY — Derives passwords from TEMP_HASH_ format
        Never returns actual PasswordHash field
        
    Password Derivation Logic:
        - IF PasswordHash starts with "TEMP_HASH_" → extract password
        - ELSE → test_password = None (not a test account)
    """
    conn = None
    
    try:
        # Open database connection
        conn = get_connection()
        
        # Get raw user data from database
        rows = get_all_user_credentials(conn)
        
        # Convert rows to list of dictionaries with password derivation
        credentials = []
        
        for row in rows:
            # TEST MODE PASSWORD DERIVATION — DO NOT USE IN PRODUCTION
            # Extract test password from TEMP_HASH_ format
            password_hash = row.PasswordHash
            test_password: Optional[str] = None
            
            if password_hash and password_hash.startswith("TEMP_HASH_"):
                # Strip TEMP_HASH_ prefix to get actual test password
                test_password = password_hash.replace("TEMP_HASH_", "")
            
            # Build credential dictionary
            credential = {
                "user_id": row.UserID,
                "username": row.Username,
                "display_name": row.DisplayName if hasattr(row, 'DisplayName') and row.DisplayName else None,
                "role": row.RoleCode if hasattr(row, 'RoleCode') and row.RoleCode else None,
                "org_unit": row.org_unit_name if hasattr(row, 'org_unit_name') and row.org_unit_name else None,
                "org_unit_id": row.OrgUnitID if hasattr(row, 'OrgUnitID') and row.OrgUnitID else None,
                "org_unit_type": row.OrgUnitType if hasattr(row, 'OrgUnitType') and row.OrgUnitType else None,
                "active": bool(row.IsActive),
                "test_password": test_password  # Derived from TEMP_HASH_, not actual hash
            }
            
            credentials.append(credential)

        # SOFTWARE_ADMIN is reserved as a technical/break-glass account and must not
        # appear in operational credential exports.
        credentials = [
            c for c in credentials
            if not (
                str(c.get("username", "")).lower() == "software_admin"
                or str(c.get("role", "")).upper() == "SOFTWARE_ADMIN"
            )
        ]

        return credentials
        
    finally:
        # Always close connection
        if conn:
            conn.close()
