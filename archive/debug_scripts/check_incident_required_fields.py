"""Check required fields in APP_IncidentCase table"""
import sys
import os

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

import pyodbc
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT COLUMN_NAME, IS_NULLABLE, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_IncidentCase' AND IS_NULLABLE = 'NO' 
    ORDER BY ORDINAL_POSITION
""")

cols = cursor.fetchall()
print("Required (NOT NULL) columns in APP_IncidentCase:")
print("=" * 70)
for col in cols:
    col_name, is_nullable, data_type, max_length = col
    length_str = f"({max_length})" if max_length else ""
    print(f"  {col_name}: {data_type}{length_str}")

conn.close()
