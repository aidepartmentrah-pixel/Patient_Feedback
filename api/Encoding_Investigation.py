import pyodbc

# -----------------------
# 1. Connect to your SQL Server
# -----------------------
def get_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 18 for SQL Server};"
        "Server=SOCIALMEDIA;"
        "Database=IncidentManager;"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
    )

conn = get_connection()
cursor = conn.cursor()

# -----------------------
# 2. Helper function to get mapping from Parameter table
# -----------------------
def get_parameter_mapping(parent_description):
    """
    Returns a dict mapping Description -> UniqueID for a given parent parameter
    """
    # First, get parent UniqueID
    cursor.execute("""
        SELECT UniqueID
        FROM dbo.Parameter
        WHERE Description = ?
    """, (parent_description,))
    row = cursor.fetchone()
    if not row:
        return {}
    parent_id = row.UniqueID

    # Now get child parameters
    cursor.execute("""
        SELECT UniqueID, Description
        FROM dbo.Parameter
        WHERE ParentID = ?
        ORDER BY Description
    """, (parent_id,))
    mapping = {desc: uid for uid, desc in cursor.fetchall()}
    return mapping

# -----------------------
# 3. Department / Section mapping
# -----------------------
cursor.execute("""
    SELECT UniqueID, Name
    FROM dbo.AdminsrationUnit
    ORDER BY Name
""")
DEPARTMENT_MAP = {name: uid for uid, name in cursor.fetchall()}

# -----------------------
# 4. Other mappings
# -----------------------
SEVERITY_MAP = get_parameter_mapping("Severity")
STATUS_MAP = get_parameter_mapping("IncidentStatus")
STAGE_MAP = get_parameter_mapping("Stage")
HARM_MAP = get_parameter_mapping("Harm")
FEEDBACK_TYPE_MAP = get_parameter_mapping("IncidentRequesterType")
SOURCE_MAP = get_parameter_mapping("IncidentSource")

# -----------------------
# 5. Print all mappings
# -----------------------
print("=== Departments / Sections ===")
print(DEPARTMENT_MAP)
print("\n=== Severity ===")
print(SEVERITY_MAP)
print("\n=== Status ===")
print(STATUS_MAP)
print("\n=== Stage ===")
print(STAGE_MAP)
print("\n=== Harm ===")
print(HARM_MAP)
print("\n=== Feedback Type ===")
print(FEEDBACK_TYPE_MAP)
print("\n=== Source ===")
print(SOURCE_MAP)

# -----------------------
# 6. Close connection
# -----------------------
cursor.close()
conn.close()
