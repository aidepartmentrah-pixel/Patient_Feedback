"""
User Inventory Database Layer
Read-only queries for organizational unit and user mapping.

Phase 5 - Module 5.1: Inventory & Mapping Engine
Provides visibility into which org units have user accounts assigned.
"""

from typing import List, Dict, Any
from core.database import get_connection


def get_org_unit_user_inventory() -> List[Dict[str, Any]]:
    """
    Get comprehensive inventory of all organizational units and their assigned users.
    
    This is a READ-ONLY query that shows:
    - All organizational units (Administrations, Departments, Sections)
    - Which users (if any) are assigned to each unit
    - The roles assigned to those users
    - User active status
    
    Uses LEFT JOINs to include org units even if they have no users assigned.
    
    Returns:
        List of dictionaries with org unit and user information:
        [
            {
                "org_unit_id": 10,
                "org_unit_name": "قسم الطوارئ",
                "org_unit_type": "SECTION",
                "username": "sec_10_admin",
                "role_code": "SECTION_ADMIN",
                "is_active": True
            },
            ...
        ]
        
    Note:
        - Org units without users will have null username/role_code
        - Type codes: 323=ADMINISTRATION, 324=SECTION, 325=DEPARTMENT
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Query joins org units with users and their roles
        # LEFT JOINs ensure we see ALL org units, even those without users
        query = """
            SELECT 
                org.UniqueID as org_unit_id,
                org.Name as org_unit_name,
                org.Type as org_unit_type_code,
                CASE org.Type
                    WHEN 323 THEN 'ADMINISTRATION'
                    WHEN 324 THEN 'SECTION'
                    WHEN 325 THEN 'DEPARTMENT'
                    ELSE 'UNKNOWN'
                END as org_unit_type_label,
                u.Username as username,
                r.RoleCode as role_code,
                u.IsActive as is_active
            FROM dbo.AdminsrationUnit org
            LEFT JOIN dbo.APP_UserRoleScope urs 
                ON org.UniqueID = urs.OrgUnitID
            LEFT JOIN dbo.APP_Users u 
                ON urs.UserID = u.UserID
            LEFT JOIN dbo.APP_Roles r 
                ON urs.RoleID = r.RoleID
            WHERE org.Frozen = 0  -- Only active org units
            ORDER BY 
                org.Type,           -- Group by type (Administration, Department, Section)
                org.Name,           -- Then alphabetically by name
                u.Username          -- Then by username
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Convert pyodbc rows to list of dictionaries
        results = []
        for row in rows:
            results.append({
                "org_unit_id": row.org_unit_id,
                "org_unit_name": row.org_unit_name,
                "org_unit_type": row.org_unit_type_label,
                "username": row.username if row.username else None,
                "role_code": row.role_code if row.role_code else None,
                "is_active": bool(row.is_active) if row.is_active is not None else None
            })
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to retrieve user inventory: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_org_units_without_users() -> List[Dict[str, Any]]:
    """
    Get organizational units that have NO users assigned.
    
    Useful for identifying which units need user accounts created.
    
    Returns:
        List of org units without any user assignments:
        [
            {
                "org_unit_id": 25,
                "org_unit_name": "قسم الجراحة",
                "org_unit_type": "SECTION"
            },
            ...
        ]
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        query = """
            SELECT 
                org.UniqueID as org_unit_id,
                org.Name as org_unit_name,
                CASE org.Type
                    WHEN 323 THEN 'ADMINISTRATION'
                    WHEN 324 THEN 'SECTION'
                    WHEN 325 THEN 'DEPARTMENT'
                    ELSE 'UNKNOWN'
                END as org_unit_type_label
            FROM dbo.AdminsrationUnit org
            WHERE org.Frozen = 0
            AND NOT EXISTS (
                SELECT 1 
                FROM dbo.APP_UserRoleScope urs
                WHERE urs.OrgUnitID = org.UniqueID
            )
            ORDER BY org.Type, org.Name
        """
        
        cursor.execute(query)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "org_unit_id": row.org_unit_id,
                "org_unit_name": row.org_unit_name,
                "org_unit_type": row.org_unit_type_label
            })
        
        return results
        
    except Exception as e:
        raise Exception(f"Failed to retrieve org units without users: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


def get_inventory_summary() -> Dict[str, Any]:
    """
    Get summary statistics about user inventory.
    
    Returns:
        Dictionary with summary counts:
        {
            "total_org_units": 150,
            "total_users": 45,
            "administrations_with_users": 3,
            "departments_with_users": 12,
            "sections_with_users": 30,
            "org_units_without_users": 105
        }
    """
    conn = None
    cursor = None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Count total org units
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM dbo.AdminsrationUnit 
            WHERE Frozen = 0
        """)
        total_org_units = cursor.fetchone().count
        
        # Count total active users
        cursor.execute("""
            SELECT COUNT(DISTINCT UserID) as count 
            FROM dbo.APP_Users 
            WHERE IsActive = 1
        """)
        total_users = cursor.fetchone().count
        
        # Count administrations with users
        cursor.execute("""
            SELECT COUNT(DISTINCT org.UniqueID) as count
            FROM dbo.AdminsrationUnit org
            INNER JOIN dbo.APP_UserRoleScope urs ON org.UniqueID = urs.OrgUnitID
            WHERE org.Type = 323 AND org.Frozen = 0
        """)
        admin_with_users = cursor.fetchone().count
        
        # Count departments with users
        cursor.execute("""
            SELECT COUNT(DISTINCT org.UniqueID) as count
            FROM dbo.AdminsrationUnit org
            INNER JOIN dbo.APP_UserRoleScope urs ON org.UniqueID = urs.OrgUnitID
            WHERE org.Type = 325 AND org.Frozen = 0
        """)
        dept_with_users = cursor.fetchone().count
        
        # Count sections with users
        cursor.execute("""
            SELECT COUNT(DISTINCT org.UniqueID) as count
            FROM dbo.AdminsrationUnit org
            INNER JOIN dbo.APP_UserRoleScope urs ON org.UniqueID = urs.OrgUnitID
            WHERE org.Type = 324 AND org.Frozen = 0
        """)
        section_with_users = cursor.fetchone().count
        
        # Count org units without users
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM dbo.AdminsrationUnit org
            WHERE org.Frozen = 0
            AND NOT EXISTS (
                SELECT 1 FROM dbo.APP_UserRoleScope urs
                WHERE urs.OrgUnitID = org.UniqueID
            )
        """)
        units_without_users = cursor.fetchone().count
        
        return {
            "total_org_units": total_org_units,
            "total_users": total_users,
            "administrations_with_users": admin_with_users,
            "departments_with_users": dept_with_users,
            "sections_with_users": section_with_users,
            "org_units_without_users": units_without_users
        }
        
    except Exception as e:
        raise Exception(f"Failed to retrieve inventory summary: {str(e)}")
    
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
