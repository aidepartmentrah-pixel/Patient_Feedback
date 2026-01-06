import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check APP_LOOKUP_CASE_STAGE columns
cursor.execute("""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_LOOKUP_CASE_STAGE'
""")

cols = cursor.fetchall()
print("APP_LOOKUP_CASE_STAGE columns:", [c[0] for c in cols])

# Check APP_IncidentCase columns
cursor.execute("""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_IncidentCase'
""")

cols = cursor.fetchall()
print("APP_IncidentCase columns:", [c[0] for c in cols])

cursor.close()
conn.close()
