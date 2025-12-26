import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SOCIALMEDIA;'
    'DATABASE=IncidentManager;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
cursor = conn.cursor()

# Get all lookup tables
print("=== ALL LOOKUP TABLES ===")
cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE='BASE TABLE' AND TABLE_NAME LIKE '%LOOKUP%' 
    ORDER BY TABLE_NAME
""")
for row in cursor.fetchall():
    print(f"  - {row[0]}")

# Check specific tables
print("\n=== DOMAIN TABLE COLUMNS ===")
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_DOMAIN")
    print(f"Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== CATEGORY TABLE COLUMNS ===")
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_CATEGORY")
    print(f"Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"Error: {e}")

print("\n=== STAGE TABLE (checking both names) ===")
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_STAGE")
    print(f"APP_LOOKUP_STAGE Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"APP_LOOKUP_STAGE Error: {e}")

try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_CASE_STAGE")
    print(f"APP_LOOKUP_CASE_STAGE Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"APP_LOOKUP_CASE_STAGE Error: {e}")

print("\n=== STATUS TABLE (checking both names) ===")
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_STATUS")
    print(f"APP_LOOKUP_STATUS Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"APP_LOOKUP_STATUS Error: {e}")

try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_CASE_STATUS")
    print(f"APP_LOOKUP_CASE_STATUS Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"APP_LOOKUP_CASE_STATUS Error: {e}")

print("\n=== SEVERITY TABLE (checking various names) ===")
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_SEVERITY")
    print(f"APP_LOOKUP_SEVERITY Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"APP_LOOKUP_SEVERITY Error: {e}")

try:
    cursor.execute("SELECT TOP 1 * FROM dbo.SeverityLevel")
    print(f"SeverityLevel Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"SeverityLevel Error: {e}")

print("\n=== HARM LEVEL TABLE ===")
try:
    cursor.execute("SELECT TOP 1 * FROM dbo.APP_LOOKUP_HARM_LEVEL")
    print(f"Columns: {[col[0] for col in cursor.description]}")
except Exception as e:
    print(f"Error: {e}")

conn.close()
