import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SOCIALMEDIA;'
    'DATABASE=IncidentManager;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
cursor = conn.cursor()

# Check Season table structure
cursor.execute("""
    SELECT 
        COLUMN_NAME, 
        DATA_TYPE, 
        IS_NULLABLE,
        COLUMNPROPERTY(OBJECT_ID('dbo.Season'), COLUMN_NAME, 'IsIdentity') as IsIdentity
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'Season' 
    ORDER BY ORDINAL_POSITION
""")

print("Season Table Structure:")
print(f"{'Column':<20} | {'Type':<15} | {'Nullable':<10} | {'Identity'}")
print('-'*70)
for row in cursor.fetchall():
    print(f"{row.COLUMN_NAME:<20} | {row.DATA_TYPE:<15} | {row.IS_NULLABLE:<10} | {row.IsIdentity}")

# Get max UniqueID
cursor.execute("SELECT MAX(UniqueID) as MaxID FROM dbo.Season")
max_id = cursor.fetchone().MaxID
print(f"\nCurrent MAX(UniqueID): {max_id}")

conn.close()
