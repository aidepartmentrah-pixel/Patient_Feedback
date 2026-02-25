"""
Execute Phase G-B1 SQL Migration Script
Creates the APP_DrawerNote table.
"""
import pyodbc

# Read SQL script
with open('backend/database_migrations/phase_g_b1_create_drawer_note_table.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()

# Connect to database
conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()

print("="*80)
print("Executing Phase G-B1 SQL Migration Script")
print("="*80)

# Split by GO and execute each batch
batches = [batch.strip() for batch in sql_script.split('GO') 
           if batch.strip() and not batch.strip().startswith('/*') or 'CREATE TABLE' in batch]

for i, batch in enumerate(batches, 1):
    if batch:
        try:
            cursor.execute(batch)
            conn.commit()
            print(f"✓ Batch {i} executed successfully")
        except Exception as e:
            print(f"✗ Batch {i} failed: {str(e)}")
            if "already exists" not in str(e):
                raise

conn.close()
print("="*80)
print("✅ Migration completed successfully")
print("="*80)
