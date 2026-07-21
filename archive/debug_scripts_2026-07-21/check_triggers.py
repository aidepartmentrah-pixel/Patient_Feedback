"""
Check for triggers on AdminsrationUnit table
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("\n=== Triggers on AdminsrationUnit ===")
cursor.execute("""
    SELECT 
        t.name AS trigger_name,
        t.type_desc AS trigger_type,
        t.is_disabled,
        OBJECT_DEFINITION(t.object_id) AS trigger_definition
    FROM sys.triggers t
    INNER JOIN sys.tables tab ON t.parent_id = tab.object_id
    WHERE tab.name = 'AdminsrationUnit'
""")

triggers = cursor.fetchall()
if triggers:
    for trigger in triggers:
        status = "DISABLED" if trigger[2] else "ENABLED"
        print(f"\nTrigger: {trigger[0]} ({trigger[1]}, {status})")
        print(f"Definition preview:")
        definition = trigger[3][:500] if trigger[3] else "N/A"
        print(definition)
else:
    print("No triggers found")

print("\n=== Checking if UniqueID is IDENTITY ===")
cursor.execute("""
    SELECT 
        COLUMN_NAME,
        COLUMNPROPERTY(OBJECT_ID('dbo.AdminsrationUnit'), COLUMN_NAME, 'IsIdentity') AS is_identity
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'AdminsrationUnit'
    AND COLUMN_NAME = 'UniqueID'
""")
result = cursor.fetchone()
if result:
    is_identity = "YES" if result[1] == 1 else "NO"
    print(f"UniqueID is IDENTITY: {is_identity}")

cursor.close()
conn.close()
