"""Execute check_patient_fk_constraints.sql"""
import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

# Read SQL file
with open('backend/sql_scripts/check_patient_fk_constraints.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Split by GO statements
batches = [batch.strip() for batch in sql_script.split('GO') if batch.strip() and not batch.strip().startswith('/*')]

cursor = conn.cursor()

print("Executing check_patient_fk_constraints.sql...")
print("="*70)

for i, batch in enumerate(batches, 1):
    if batch and 'PRINT' not in batch.upper():
        try:
            cursor.execute(batch)
            
            # If it's a SELECT query, fetch and display results
            if batch.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                if results:
                    print(f"\nBatch {i} results:")
                    for row in results:
                        print(f"  {row}")
                else:
                    print(f"\nBatch {i}: No results found")
        except Exception as e:
            print(f"✗ Batch {i} failed: {str(e)}")
            
conn.close()
print("\n" + "="*70)
print("FK constraint check complete!")
