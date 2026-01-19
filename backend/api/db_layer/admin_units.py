import pyodbc

def get_connection():
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=SOCIALMEDIA;"
        "DATABASE=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )
    return conn


def get_admin_unit_by_id(admin_unit_id: int):
    """
    Return one administration unit by its UniqueID.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT *
        FROM AdminsrationUnit
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
        FROM AdminsrationUnit
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
        FROM AdminsrationUnit
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
        FROM AdminsrationUnit child
        JOIN AdminsrationUnit parent
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
        FROM AdminsrationUnit
        """
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_admin_unit_leaves():
    """
    Return administration units that have no children (leaf nodes).
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT u.*
        FROM AdminsrationUnit u
        LEFT JOIN AdminsrationUnit c
            ON u.UniqueID = c.ParentID
        WHERE c.UniqueID IS NULL
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
        FROM AdminsrationUnit
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
        FROM AdminsrationUnit
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
