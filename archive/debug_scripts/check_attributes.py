import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SOCIALMEDIA;'
    'DATABASE=IncidentManager;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)

cursor = conn.cursor()

print("\n=== EXISTING ATTRIBUTE TYPES ===")
cursor.execute("SELECT AttributeType, AttributeTypeLabel, AttributeTypeLabelAr FROM dbo.APP_AttributeType")
for row in cursor.fetchall():
    print(f"{row[0]:20} | {row[1]:30} | {row[2]}")

print("\n=== CHECKING FOR SETTINGS TABLE ===")
cursor.execute("""
SELECT TABLE_NAME 
FROM INFORMATION_SCHEMA.TABLES 
WHERE TABLE_NAME LIKE '%SETTING%' OR TABLE_NAME LIKE '%CONFIG%'
ORDER BY TABLE_NAME
""")
print("Settings-related tables:")
for row in cursor.fetchall():
    print(f"  - {row[0]}")

conn.close()
