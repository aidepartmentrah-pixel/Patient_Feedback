from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Get primary key info
cursor.execute("""
SELECT kcu.COLUMN_NAME
FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu 
    ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
WHERE tc.TABLE_NAME = 'APP_IncidentCaseFeedback' 
    AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
""")
print('Primary key column(s):')
for r in cursor.fetchall():
    print(f'  {r[0]}')

# Check if RCA exists for incident 493
cursor.execute('SELECT IncidentRequestCaseID, AdministrativeSubcaseID FROM APP_IncidentCaseFeedback WHERE IncidentRequestCaseID = 493')
rows = cursor.fetchall()
print(f'\nExisting RCA for incident 493:')
for r in rows:
    print(f'  incident_id={r[0]}, subcase_id={r[1]}')

# Also check subcase 522
cursor.execute('SELECT * FROM APP_AdministrativeSubcase WHERE SubcaseID = 522')
row = cursor.fetchone()
if row:
    cols = [c[0] for c in cursor.description]
    data = dict(zip(cols, row))
    print(f'\nSubcase 522:')
    print(f'  incident_id={data.get("IncidentRequestCaseID")}')
    print(f'  status={data.get("Status")}')

conn.close()
