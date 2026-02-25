from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Find status tables
cursor.execute("""
    SELECT TABLE_NAME 
    FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_NAME LIKE '%Status%' OR TABLE_NAME LIKE '%Code%' OR TABLE_NAME LIKE '%Lookup%'
""")
print("Tables with Status/Code/Lookup:")
for r in cursor.fetchall():
    print(f"  {r[0]}")

# Check if there are existing records in feedback table 
cursor.execute("SELECT COUNT(*) FROM APP_IncidentCaseFeedback")
print(f"\nTotal feedback records: {cursor.fetchone()[0]}")

conn.close()
