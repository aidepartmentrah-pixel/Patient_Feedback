"""
DB Layer for User Credentials Listing (TEST ONLY)
Read-only queries for viewing all user accounts with role assignments.

⚠️ TEST ONLY — DO NOT USE IN PRODUCTION
"""

from typing import List, Any


def get_all_user_credentials(conn) -> List[Any]:
    """
    Get all users with their roles, org units, and password hashes.
    
    Args:
        conn: Active database connection
        
    Returns:
        List of row objects with user, role, and org unit data
        
    Note:
        READ-ONLY query
        Returns PasswordHash as-is (service layer handles password derivation)
        LEFT JOINs ensure users without scopes are included
    """
    cursor = conn.cursor()
    
    try:
        query = """
            SELECT 
                u.UserID,
                u.Username,
                u.PasswordHash,
                u.IsActive,
                r.RoleCode,
                a.Name AS org_unit_name,
                s.OrgUnitType,
                s.OrgUnitID
            FROM dbo.APP_Users u
            LEFT JOIN dbo.APP_UserRoleScope s ON u.UserID = s.UserID
            LEFT JOIN dbo.APP_Roles r ON s.RoleID = r.RoleID
            LEFT JOIN dbo.AdminsrationUnit a ON s.OrgUnitID = a.UniqueID
            ORDER BY u.UserID, r.RoleCode
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        return rows
        
    finally:
        cursor.close()
