import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

tables = [
    'APP_LOOKUP_HARM_LEVEL',
    'APP_LOOKUP_SEVERITY',
    'APP_LOOKUP_CLASSIFICATION',
    'APP_LOOKUP_CATEGORY',
    'APP_LOOKUP_SUBCATEGORY',
    'APP_LOOKUP_DOMAIN'
]

for table in tables:
    cursor.execute(f"""
        SELECT COLUMN_NAME 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = '{table}'
    """)
    cols = cursor.fetchall()
    print(f"{table} columns: {[c[0] for c in cols]}")

cursor.close()
conn.close()
