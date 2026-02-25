import sys
sys.path.insert(0, '.')

from backend.api.db_layer import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check if table has AdministrativeSubcaseID column
cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_IncidentCaseFeedback'
    ORDER BY ORDINAL_POSITION
""")

print("APP_IncidentCaseFeedback columns:")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
