import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
cursor = conn.cursor()

# Find all tables with 'Patient' in the name
cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE '%Patient%' OR TABLE_NAME LIKE '%PATIENT%'
""")

print("=== PATIENT TABLES ===")
for row in cursor.fetchall():
    print(row[0])

# Check APP_VIEWTABLE_PATIENT_ADMISSION columns
print("\n=== APP_VIEWTABLE_PATIENT_ADMISSION COLUMNS ===")
cursor.execute("SELECT TOP 1 * FROM APP_VIEWTABLE_PATIENT_ADMISSION")
for col in cursor.description:
    print(f"{col[0]}: {col[1]}")

conn.close()
