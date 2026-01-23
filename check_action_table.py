import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()

# Check if APP_ActionItem table exists
print("=== Tables with 'Action' in name ===")
cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME LIKE '%Action%'")
tables = cursor.fetchall()
for row in tables:
    print(f"  - {row[0]}")

if tables:
    # Get column structure
    print("\n=== APP_ActionItem columns ===")
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'APP_ActionItem'
        ORDER BY ORDINAL_POSITION
    """)
    columns = cursor.fetchall()
    for col in columns:
        print(f"  {col[0]}: {col[1]} (nullable: {col[2]})")
else:
    print("\nNo APP_ActionItem table found!")
    print("\n=== Available tables ===")
    cursor.execute("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'dbo' ORDER BY TABLE_NAME")
    all_tables = cursor.fetchall()
    for row in all_tables:
        print(f"  - {row[0]}")

conn.close()
