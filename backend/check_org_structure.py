"""
Check AdminsrationUnit table structure and find a valid department
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("\n=== AdminsrationUnit Table - Type Distribution ===")
cursor.execute("""
    SELECT Type, COUNT(*) as count
    FROM dbo.AdminsrationUnit
    GROUP BY Type
    ORDER BY Type
""")
type_dist = cursor.fetchall()
for row in type_dist:
    type_name = {323: 'Administration', 324: 'Section', 325: 'Department'}.get(row[0], 'Unknown')
    print(f"Type {row[0]} ({type_name}): {row[1]} units")

print("\n=== Sample Departments (Type=325) ===")
cursor.execute("""
    SELECT TOP 5 UniqueID, Name, ParentID, Frozen
    FROM dbo.AdminsrationUnit
    WHERE Type = 325
    ORDER BY UniqueID
""")
departments = cursor.fetchall()
for dept in departments:
    frozen_status = "FROZEN" if dept[3] else "Active"
    print(f"ID={dept[0]}: {dept[1]} (ParentID={dept[2]}, {frozen_status})")

print("\n=== Checking if ID=5 exists ===")
cursor.execute("""
    SELECT UniqueID, Name, Type, ParentID, Frozen
    FROM dbo.AdminsrationUnit
    WHERE UniqueID = 5
""")
result = cursor.fetchone()
if result:
    type_name = {323: 'Administration', 324: 'Section', 325: 'Department'}.get(result[2], 'Unknown')
    frozen_status = "FROZEN" if result[4] else "Active"
    print(f"✓ ID 5 exists: {result[1]}")
    print(f"  Type: {result[2]} ({type_name})")
    print(f"  ParentID: {result[3]}")
    print(f"  Status: {frozen_status}")
else:
    print("✗ ID 5 does NOT exist")

cursor.close()
conn.close()
