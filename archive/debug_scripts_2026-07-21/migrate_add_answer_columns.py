"""
Migration: Add ShowSectionAnswer, ShowDepartmentAnswer, ShowAdministrationAnswer
to dbo.APP_CUSTOM_VIEWS table.
"""
import sys
sys.path.insert(0, ".")
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("""
    SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = 'APP_CUSTOM_VIEWS'
    AND COLUMN_NAME IN ('ShowSectionAnswer', 'ShowDepartmentAnswer', 'ShowAdministrationAnswer')
""")
existing = {row[0] for row in cursor.fetchall()}
print("Already existing:", existing)

to_add = [
    "ShowSectionAnswer",
    "ShowDepartmentAnswer",
    "ShowAdministrationAnswer",
]

for col in to_add:
    if col not in existing:
        cursor.execute(f"ALTER TABLE dbo.APP_CUSTOM_VIEWS ADD {col} BIT NOT NULL DEFAULT 0")
        print(f"Added column: {col}")
    else:
        print(f"Skipped (already exists): {col}")

conn.commit()
conn.close()
print("Migration complete.")
