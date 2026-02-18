import sys
sys.path.append('c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend')

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check what Type values exist
cursor.execute("""
    SELECT DISTINCT Type, COUNT(*) as count
    FROM AdminsrationUnit
    WHERE Frozen = 0
    GROUP BY Type
    ORDER BY Type
""")

print("\n=== Type Values in Database ===")
for row in cursor.fetchall():
    print(f"Type {row[0]}: {row[1]} units")

# Check some sample names for each type
cursor.execute("""
    SELECT Type, Name
    FROM AdminsrationUnit
    WHERE Frozen = 0
    ORDER BY Type
""")

print("\n=== Sample Names by Type ===")
current_type = None
count = 0
for row in cursor.fetchall():
    if current_type != row[0]:
        current_type = row[0]
        count = 0
        print(f"\nType {row[0]}:")
    if count < 3:
        print(f"  - {row[1]}")
        count += 1

conn.close()
