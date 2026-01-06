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
    Return administration units that are not frozen.
    Returns list of dicts with UniqueID and Name.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT UniqueID, Name
        FROM AdminsrationUnit
        WHERE Frozen = 0
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
