import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_IncidentCaseTargetDepartment'
""")

cols = cursor.fetchall()
print('APP_IncidentCaseTargetDepartment columns:', [c[0] for c in cols])

cursor.close()
conn.close()
