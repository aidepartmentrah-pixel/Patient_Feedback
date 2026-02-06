"""
Query actual APP_AdministrativeSubcase schema
"""
import sys
import os

backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

print("\n" + "="*80)
print("ACTUAL APP_AdministrativeSubcase SCHEMA")
print("="*80)

cursor.execute("""
    SELECT 
        COLUMN_NAME,
        DATA_TYPE,
        CHARACTER_MAXIMUM_LENGTH,
        IS_NULLABLE
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'APP_AdministrativeSubcase'
    ORDER BY ORDINAL_POSITION
""")

columns = cursor.fetchall()

print(f"\nTable: APP_AdministrativeSubcase")
print(f"Total columns: {len(columns)}\n")
print(f"{'Column Name':<40} {'Data Type':<20} {'Length':<10} {'Nullable'}")
print("="*80)

for col in columns:
    col_name = col.COLUMN_NAME
    data_type = col.DATA_TYPE
    max_len = col.CHARACTER_MAXIMUM_LENGTH if col.CHARACTER_MAXIMUM_LENGTH else ''
    nullable = col.IS_NULLABLE
    print(f"{col_name:<40} {data_type:<20} {str(max_len):<10} {nullable}")

# Also show the actual data in SubcaseID 53
print("\n" + "="*80)
print("DATA IN SubcaseID 53")
print("="*80)

# Get all column names
col_names = [col.COLUMN_NAME for col in columns]

# Build dynamic query
query = f"SELECT {', '.join(col_names)} FROM APP_AdministrativeSubcase WHERE SubcaseID = 53"
cursor.execute(query)
row = cursor.fetchone()

if row:
    print("\nSubcaseID 53 actual values:")
    for i, col_name in enumerate(col_names):
        value = getattr(row, col_name)
        print(f"  {col_name}: {value}")
else:
    print("\nSubcaseID 53 not found!")

cursor.close()
conn.close()

print("\n" + "="*80)
