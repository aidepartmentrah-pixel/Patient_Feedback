from core.database import get_connection
conn = get_connection()
cursor = conn.cursor()

# Find user tables
cursor.execute("""SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE '%ser%' OR TABLE_NAME LIKE '%auth%' ORDER BY TABLE_NAME""")
print("User/Auth tables:", [r[0] for r in cursor.fetchall()])

# Get columns for Users table 
cursor.execute("""SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'Users' ORDER BY ORDINAL_POSITION""")
print("Users columns:", [r[0] for r in cursor.fetchall()])

# Find admin user
cursor.execute("SELECT TOP 3 * FROM dbo.Users WHERE IsAdmin = 1")
cols = [d[0] for d in cursor.description]
print(f"Users columns: {cols}")
for r in cursor.fetchall():
    print(list(r))

conn.close()
