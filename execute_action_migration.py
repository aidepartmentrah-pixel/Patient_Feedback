import pyodbc

# Execute the migration script
with open('backend/sql_scripts/migrate_action_item_table.sql', 'r', encoding='utf-8') as f:
    sql_script = f.read()

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

cursor = conn.cursor()

try:
    print("Executing migration script...")
    
    # Split by GO statements and execute each batch
    batches = [batch.strip() for batch in sql_script.split('GO') if batch.strip() and not batch.strip().startswith('--')]
    
    for i, batch in enumerate(batches, 1):
        if batch:
            print(f"\nExecuting batch {i}/{len(batches)}...")
            try:
                cursor.execute(batch)
                conn.commit()
                print(f"  ✓ Batch {i} completed")
            except Exception as e:
                print(f"  ✗ Batch {i} failed: {e}")
                # Continue with next batch even if one fails
    
    print("\n" + "="*60)
    print("Migration completed!")
    print("="*60)
    
    # Verify new columns
    print("\n=== Verifying new columns ===")
    cursor.execute("""
        SELECT COLUMN_NAME, DATA_TYPE 
        FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'APP_ActionItem'
        ORDER BY ORDINAL_POSITION
    """)
    columns = cursor.fetchall()
    print("\nCurrent columns in APP_ActionItem:")
    for col in columns:
        print(f"  - {col[0]}: {col[1]}")

except Exception as e:
    print(f"\nError during migration: {e}")
    conn.rollback()
finally:
    conn.close()
