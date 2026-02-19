from core.database import get_connection


def get_admin_unit_by_id(admin_unit_id: int):
    """
    Return one administration unit by its UniqueID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM AdminsrationUnit WITH (NOLOCK)
        WHERE UniqueID = ?
        """,
        admin_unit_id
    )

    row = cursor.fetchone()
    conn.close()
    return row


def get_admin_unit_type(admin_unit_id: int) -> int | None:
    """
    Get the Type of an administration unit by its UniqueID.
    
    Args:
        admin_unit_id: UniqueID of the organizational unit
    
    Returns:
        Type value (int) or None if not found
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT Type
        FROM AdminsrationUnit WITH (NOLOCK)
        WHERE UniqueID = ?
        """,
        admin_unit_id
    )

    row = cursor.fetchone()
    conn.close()
    
    if row:
        return row.Type
    return None


def get_admin_unit_children(parent_id: int):
    """
    Return direct children of a given administration unit.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM AdminsrationUnit WITH (NOLOCK)
        WHERE ParentID = ?
        """,
        parent_id
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_admin_unit_parent(admin_unit_id: int):
    """
    Return the parent administration unit of a given unit.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT parent.*
        FROM AdminsrationUnit child WITH (NOLOCK)
        JOIN AdminsrationUnit parent WITH (NOLOCK)
            ON child.ParentID = parent.UniqueID
        WHERE child.UniqueID = ?
        """,
        admin_unit_id
    )

    row = cursor.fetchone()
    conn.close()
    return row


def get_admin_unit_tree():
    """
    Return all administration units.
    Tree construction is done in the service layer.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM AdminsrationUnit WITH (NOLOCK)
        """
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_admin_unit_leaves():
    """
    Return administration units that have no children (leaf nodes).
    Excludes frozen units and units with NULL/empty names.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT u.*
        FROM AdminsrationUnit u WITH (NOLOCK)
        LEFT JOIN AdminsrationUnit c WITH (NOLOCK)
            ON u.UniqueID = c.ParentID
        WHERE c.UniqueID IS NULL
          AND (u.Frozen = 0 OR u.Frozen IS NULL)
          AND u.Name IS NOT NULL
          AND u.Name <> ''
          AND u.Name <> 'NULL'
        ORDER BY u.Name
        """
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_active_admin_units():
    """
    Return administration units that are not frozen and have valid type.
    Returns list of dicts with UniqueID and Name.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT UniqueID, Name
        FROM AdminsrationUnit WITH (NOLOCK)
        WHERE Frozen = 0 AND Type IS NOT NULL
        """
    )

    rows = cursor.fetchall()
    conn.close()
    
    # Convert pyodbc.Row objects to dicts
    result = []
    for row in rows:
        result.append({
            "UniqueID": row[0],
            "Name": row[1]
        })
    return result


def get_units_by_type(unit_type: int):
    """
    Get all active organizational units of a specific type.
    Excludes frozen units and units with NULL type.
    
    Args:
        unit_type: 323=Administration, 324=Section, 325=Department
        
    Returns:
        List of dicts with UniqueID and Name
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT UniqueID, Name, ParentID
        FROM AdminsrationUnit WITH (NOLOCK)
        WHERE Frozen = 0 AND Type = ? AND Type IS NOT NULL
        ORDER BY Name
        """,
        unit_type
    )

    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for row in rows:
        result.append({
            "id": row[0],
            "name": row[1],
            "parent_id": row[2]
        })
    return result


def get_unit_hierarchy(unit_id: int) -> dict:
    """
    Get the full hierarchy for a unit (self, parent, grandparent).
    
    Returns the unit's name, its parent's name (department for sections),
    and grandparent's name (administration for sections, or parent administration for departments).
    
    Args:
        unit_id: The organizational unit ID
        
    Returns:
        Dict with:
        - unit_id, unit_name, unit_type
        - parent_id, parent_name (if exists)
        - grandparent_id, grandparent_name (if exists)
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Get unit and its parent/grandparent in one query
        cursor.execute(
            """
            SELECT 
                u.UniqueID as unit_id,
                u.Name as unit_name,
                u.Type as unit_type,
                u.ParentID as parent_id,
                p.Name as parent_name,
                p.Type as parent_type,
                p.ParentID as grandparent_id,
                gp.Name as grandparent_name,
                gp.Type as grandparent_type
            FROM AdminsrationUnit u WITH (NOLOCK)
            LEFT JOIN AdminsrationUnit p WITH (NOLOCK) ON u.ParentID = p.UniqueID
            LEFT JOIN AdminsrationUnit gp WITH (NOLOCK) ON p.ParentID = gp.UniqueID
            WHERE u.UniqueID = ?
            """,
            unit_id
        )
        
        row = cursor.fetchone()
        if not row:
            return None
        
        return {
            "unit_id": row.unit_id,
            "unit_name": row.unit_name,
            "unit_type": row.unit_type,
            "parent_id": row.parent_id,
            "parent_name": row.parent_name,
            "parent_type": row.parent_type,
            "grandparent_id": row.grandparent_id,
            "grandparent_name": row.grandparent_name,
            "grandparent_type": row.grandparent_type
        }
    finally:
        cursor.close()
        conn.close()
