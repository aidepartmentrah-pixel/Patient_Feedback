from core.database import get_connection
conn = get_connection()
cursor = conn.cursor()

# Check APP_Users columns
cursor.execute("""SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'APP_Users' ORDER BY ORDINAL_POSITION""")
print("APP_Users columns:", [r[0] for r in cursor.fetchall()])

# Check Users table - use SELECT *
cursor.execute("SELECT TOP 3 * FROM dbo.Users")
cols = [d[0] for d in cursor.description]
print(f"\nUsers columns: {cols}")
for r in cursor.fetchall():
    vals = list(r)
    # Mask password
    if 'Password' in cols:
        idx = cols.index('Password')
        vals[idx] = '***'
    print(dict(zip(cols, vals)))

# Check login endpoint
import os, glob
# Find auth files
for f in glob.glob("api/**/auth*", recursive=True):
    print(f"\nAuth file: {f}")

conn.close()
