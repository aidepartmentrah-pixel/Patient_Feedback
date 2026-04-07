"""Quick diagnostic script to check database data."""
import pyodbc
from core.config_loader import get_config

config = get_config()
db = config['database']
conn_str = f"DRIVER={{{db['driver']}}};SERVER={db['server']};DATABASE={db['database']};UID={db['username']};PWD={db['password']};TrustServerCertificate=yes"

print(f"Connecting to: {db['server']}/{db['database']}")
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# List all tables first
print('\n=== ALL TABLES IN DATABASE ===')
cursor.execute("""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES 
    WHERE TABLE_TYPE = 'BASE TABLE' 
    ORDER BY TABLE_NAME
""")
tables = [row[0] for row in cursor.fetchall()]
for t in tables:
    print(f'  {t}')

# Check incident counts - try common table names
print('\n=== DATA COUNTS ===')
for table_name in ['IncidentCase', 'Incident_Case', 'incident_case', 'incidents', 'Incidents', 'tbl_IncidentCase']:
    try:
        cursor.execute(f'SELECT COUNT(*) FROM [{table_name}]')
        total = cursor.fetchone()[0]
        print(f'{table_name}: {total} records')
        break
    except:
        pass

# Check APP_ tables which are the actual application tables
cursor.execute('SELECT COUNT(*) FROM APP_IncidentCase')
incidents = cursor.fetchone()[0]
print(f'APP_IncidentCase: {incidents} records')

cursor.execute('SELECT COUNT(*) FROM APP_Users')
users = cursor.fetchone()[0]
print(f'APP_Users: {users} records')

# Check for patients in VW_PatientAdmission
try:
    cursor.execute('SELECT COUNT(*) FROM VW_PatientAdmission')
    patients = cursor.fetchone()[0]
    print(f'VW_PatientAdmission: {patients} records')
except Exception as e:
    print(f'VW_PatientAdmission error: {e}')

# Show sample incident data
if incidents > 0:
    print(f'\n=== LATEST 5 INCIDENTS ===')
    cursor.execute('SELECT TOP 5 id, created_at, case_stage_id, case_status_id FROM APP_IncidentCase ORDER BY id DESC')
    for row in cursor.fetchall():
        print(f'  ID: {row[0]}, Created: {row[1]}, Stage: {row[2]}, Status: {row[3]}')
else:
    print('\n*** NO INCIDENT DATA IN APP_IncidentCase ***')

# Show users
print(f'\n=== APP USERS ===')
cursor.execute('SELECT id, username, role_code FROM APP_Users')
for row in cursor.fetchall():
    print(f'  ID: {row[0]}, Username: {row[1]}, Role: {row[2]}')

conn.close()
print('\n=== DIAGNOSIS COMPLETE ===')
