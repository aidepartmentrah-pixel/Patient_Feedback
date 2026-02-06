"""
Quick script to check APP_Users table status
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Count users
cursor.execute("SELECT COUNT(*) FROM APP_Users")
count = cursor.fetchone()[0]
print(f"\n=== APP_Users Table ===")
print(f"Total users: {count}")

# List all usernames
cursor.execute("SELECT UserID, Username FROM APP_Users ORDER BY UserID")
users = cursor.fetchall()
print(f"\nAll users:")
for user in users:
    print(f"  UserID={user[0]}: {user[1]}")

cursor.close()
conn.close()
