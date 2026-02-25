"""Execute create_reserve_patient_table.sql"""
import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# Read SQL file
with open('backend/sql_scripts/create_reserve_patient_table.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Split by GO statements
batches = [batch.strip() for batch in sql_script.split('GO') if batch.strip() and not batch.strip().startswith('/*')]

cursor = conn.cursor()

print("Executing create_reserve_patient_table.sql...")
print("="*70)

for i, batch in enumerate(batches, 1):
    if batch:
        try:
            cursor.execute(batch)
            conn.commit()
            print(f"✓ Batch {i} executed successfully")
        except Exception as e:
            print(f"✗ Batch {i} failed: {str(e)}")
            
conn.close()
print("="*70)
print("SQL script execution complete!")
