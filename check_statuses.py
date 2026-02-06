import pyodbc

conn = pyodbc.connect(
    'DRIVER={ODBC Driver 17 for SQL Server};'
    'SERVER=SOCIALMEDIA;'
    'DATABASE=IncidentManager;'
    'Trusted_Connection=yes;'
    'TrustServerCertificate=yes;'
)
cursor = conn.cursor()
cursor.execute('SELECT DISTINCT Status FROM dbo.APP_AdministrativeSubcase ORDER BY Status')
rows = cursor.fetchall()
print('Currently Used Status Values:')
for row in rows:
    print(f'  - {row.Status}')
cursor.close()
conn.close()
