import pyodbc

conn = pyodbc.connect(
    "DRIVER={ODBC Driver 17 for SQL Server};"
    "SERVER=SOCIALMEDIA;"
    "DATABASE=IncidentManager;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)
cursor = conn.cursor()

print("APP_VIEWTABLE_PATIENT_ADMISSION columns:")
cursor.execute("""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_VIEWTABLE_PATIENT_ADMISSION'
    ORDER BY ORDINAL_POSITION
""")
hospital_cols = [row[0] for row in cursor.fetchall()]
print(f"Total: {len(hospital_cols)}")
for col in hospital_cols:
    print(f"  - {col}")

print("\n" + "="*70 + "\n")

print("APP_RESERVE_PATIENT columns:")
cursor.execute("""
    SELECT COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_RESERVE_PATIENT'
    ORDER BY ORDINAL_POSITION
""")
reserve_cols = [row[0] for row in cursor.fetchall()]
print(f"Total: {len(reserve_cols)}")
for col in reserve_cols:
    print(f"  - {col}")

print("\n" + "="*70 + "\n")

# Find differences
extra_in_reserve = set(reserve_cols) - set(hospital_cols)
missing_in_reserve = set(hospital_cols) - set(reserve_cols)

if extra_in_reserve:
    print(f"Extra columns in RESERVE (not in hospital):")
    for col in extra_in_reserve:
        print(f"  - {col}")

if missing_in_reserve:
    print(f"Missing columns in RESERVE (in hospital but not reserve):")
    for col in missing_in_reserve:
        print(f"  - {col}")

if not extra_in_reserve and not missing_in_reserve:
    print("✓ All columns match perfectly!")

conn.close()
