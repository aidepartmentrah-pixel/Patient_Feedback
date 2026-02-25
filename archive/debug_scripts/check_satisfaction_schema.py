"""Check APP_IncidentCaseSatisfaction table schema"""
import sys
sys.path.insert(0, 'backend')

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_IncidentCaseSatisfaction'
    ORDER BY ORDINAL_POSITION
""")

print("APP_IncidentCaseSatisfaction columns:")
for row in cursor.fetchall():
    print(f"  - {row[0]}: {row[1]}" + (f"({row[2]})" if row[2] else ""))

cursor.close()
conn.close()
