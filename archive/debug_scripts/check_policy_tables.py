"""Check for policy tables"""
import sys
sys.path.insert(0, "c:\\Users\\IT\\Documents\\GitHub Repository\\Patient_Feedback\\backend")

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE '%Policy%' OR TABLE_NAME LIKE '%Threshold%'
""")

print("Policy-related tables:")
for row in cursor.fetchall():
    print(f"  - {row[0]}")

conn.close()
