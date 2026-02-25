import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

cursor.execute("SELECT DISTINCT Status FROM APP_AdministrativeSubcase")

print("\nSubcase statuses in database:")
for row in cursor.fetchall():
    print(f"  - '{row[0]}'")

cursor.close()
conn.close()
