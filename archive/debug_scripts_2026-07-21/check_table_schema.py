"""
Check AdminsrationUnit table schema
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("\n=== AdminsrationUnit Table Schema ===")
cursor.execute("""
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        IS_NULLABLE,
        COLUMN_DEFAULT
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'AdminsrationUnit'
    ORDER BY ORDINAL_POSITION
""")

columns = cursor.fetchall()
for col in columns:
    nullable = "NULL" if col[2] == "YES" else "NOT NULL"
    default = f" DEFAULT {col[3]}" if col[3] else ""
    print(f"{col[0]:<20} {col[1]:<15} {nullable}{default}")

cursor.close()
conn.close()
