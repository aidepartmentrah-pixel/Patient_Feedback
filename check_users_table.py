"""Check APP_Users table structure"""
from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check APP_Users columns
print("\n=== APP_Users table columns ===")
cursor.execute("""
    SELECT TOP 1 * FROM dbo.APP_Users
""")

for col in cursor.description:
    print(f"  - {col[0]}")

# Check a sample user
cursor.execute("""
    SELECT TOP 1
        UserID,
        Username
    FROM dbo.APP_Users
""")

row = cursor.fetchone()
if row:
    print(f"\nSample user: UserID={row.UserID}, Username={row.Username}")

cursor.close()
conn.close()
