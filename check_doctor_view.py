"""Quick check of APP_VIEWTABLE_VW_DOCTORS view"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("="*70)
print("Checking APP_VIEWTABLE_VW_DOCTORS View")
print("="*70)

# Check if it's a view or table
cursor.execute("""
    SELECT 
        OBJECT_NAME(object_id) as name,
        type_desc
    FROM sys.objects
    WHERE name = 'APP_VIEWTABLE_VW_DOCTORS'
""")
obj_info = cursor.fetchone()
if obj_info:
    print(f"\nObject Type: {obj_info.type_desc}")
else:
    print("\nObject not found!")

# Check what it selects from
cursor.execute("""
    SELECT TOP 1 *
    FROM APP_VIEWTABLE_VW_DOCTORS
    ORDER BY DoctorID DESC
""")
columns = [col[0] for col in cursor.description]
print(f"\nColumns: {', '.join(columns)}")

row = cursor.fetchone()
if row:
    print(f"\nLast Doctor Record:")
    for col, val in zip(columns, row):
        print(f"  {col}: {val}")

# Check if there's a trigger on reserve table
cursor.execute("""
    SELECT 
        name,
        type_desc
    FROM sys.triggers
    WHERE parent_id = OBJECT_ID('APP_RESERVE_DOCTOR')
""")
triggers = cursor.fetchall()
if triggers:
    print(f"\nTriggers on APP_RESERVE_DOCTOR:")
    for trig in triggers:
        print(f"  - {trig.name} ({trig.type_desc})")
else:
    print(f"\nNo triggers on APP_RESERVE_DOCTOR")

cursor.close()
conn.close()
