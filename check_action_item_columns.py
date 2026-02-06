"""Check columns in APP_SubcaseActionItem table"""
import sys
import os

# Add backend directory to Python path
backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_SubcaseActionItem' 
    ORDER BY ORDINAL_POSITION
""")

cols = cursor.fetchall()
print("Columns in APP_SubcaseActionItem:")
print("=" * 70)
for col in cols:
    col_name, data_type, is_nullable = col
    nullable_str = "NULL" if is_nullable == 'YES' else "NOT NULL"
    print(f"  {col_name}: {data_type} ({nullable_str})")

conn.close()
