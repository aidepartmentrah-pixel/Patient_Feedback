"""Quick check for patients in database"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# Check hospital patients
cursor.execute("SELECT TOP 5 PatientAdmissionID, FullName, MedicalFileNumber FROM dbo.APP_VIEWTABLE_PATIENT_ADMISSION")
hospital = cursor.fetchall()

print(f"Hospital patients: {len(hospital)}")
for row in hospital:
    print(f"  • ID={row[0]}, Name={row[1]}, MRN={row[2]}")

# Check reserve patients
cursor.execute("SELECT TOP 5 PatientAdmissionID, FullName, MedicalFileNumber FROM dbo.APP_RESERVE_PATIENT")
reserve = cursor.fetchall()

print(f"\nReserve patients: {len(reserve)}")
for row in reserve:
    print(f"  • ID={row[0]}, Name={row[1]}, MRN={row[2]}")

conn.close()
