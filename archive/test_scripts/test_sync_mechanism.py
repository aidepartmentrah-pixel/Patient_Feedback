"""Test if reserve doctors actually appear in APP_VIEWTABLE_VW_DOCTORS"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from backend.core.database import get_connection
from backend.api.db_layer.doctors_db import create_doctor
from datetime import datetime

# Create a test reserve doctor
test_name = f"Dr. Sync Test {datetime.now().strftime('%Y%m%d_%H%M%S')}"
print(f"Creating reserve doctor: {test_name}")
result = create_doctor(
    doctor_name=test_name,
    specialty="Sync Test",
    is_active=True,
    source_system="SYNC_TEST"
)
doctor_id = result['id']
print(f"Created with ID: {doctor_id}")

# Check if it appears in APP_VIEWTABLE_VW_DOCTORS
conn = get_connection()
cursor = conn.cursor()

print(f"\nChecking APP_VIEWTABLE_VW_DOCTORS...")
cursor.execute(
    "SELECT COUNT(*), MAX(DoctorID) FROM APP_VIEWTABLE_VW_DOCTORS WHERE DoctorID = ?",
    (doctor_id,)
)
count, found_id = cursor.fetchone()
print(f"Count in APP_VIEWTABLE_VW_DOCTORS: {count}")

if count > 0:
    print(f"✓ Reserve doctor IS in APP_VIEWTABLE_VW_DOCTORS!")
    cursor.execute("SELECT * FROM APP_VIEWTABLE_VW_DOCTORS WHERE DoctorID = ?", (doctor_id,))
    row = cursor.fetchone()
    if row:
        print(f"  Name: {row.Name}")
        print(f"  SpecialityID: {row.SpecialityID}")
        print(f"  IsActive: {row.IsActive}")
else:
    print(f"✗ Reserve doctor NOT in APP_VIEWTABLE_VW_DOCTORS")
    print(f"  This means incident validation will REJECT reserve doctors!")

# Check APP_RESERVE_DOCTOR
cursor.execute("SELECT * FROM APP_RESERVE_DOCTOR WHERE DoctorID = ?", (doctor_id,))
row = cursor.fetchone()
if row:
    print(f"\n✓ Doctor IS in APP_RESERVE_DOCTOR:")
    print(f"  DoctorID: {row.DoctorID}")
    print(f"  DoctorName: {row.DoctorName}")
    print(f"  Specialty: {row.Specialty}")

# Cleanup
print(f"\nCleaning up test doctor...")
cursor.execute("DELETE FROM APP_RESERVE_DOCTOR WHERE DoctorID = ?", (doctor_id,))
conn.commit()

# Check again after delete
cursor.execute("SELECT COUNT(*) FROM APP_VIEWTABLE_VW_DOCTORS WHERE DoctorID = ?", (doctor_id,))
count_after = cursor.fetchone()[0]
print(f"Count in APP_VIEWTABLE_VW_DOCTORS after delete: {count_after}")

if count_after == 0 and count > 0:
    print("✓ Confirmed: There IS a sync mechanism (trigger or view refresh)")
elif count > 0 and count_after > 0:
    print("⚠ Orphaned record in APP_VIEWTABLE_VW_DOCTORS")
    # Clean up orphan
    cursor.execute("DELETE FROM APP_VIEWTABLE_VW_DOCTORS WHERE DoctorID = ?", (doctor_id,))
    conn.commit()
    print("  Cleaned up orphan")

cursor.close()
conn.close()

print("\n" + "="*70)
print("CONCLUSION:")
if count > 0:
    print("Reserve doctors DO appear in APP_VIEWTABLE_VW_DOCTORS")
    print("Incident validation will ACCEPT reserve doctors ✓")
else:
    print("Reserve doctors DO NOT appear in APP_VIEWTABLE_VW_DOCTORS")
    print("insert_service.py needs to be updated to use UNION query!")
print("="*70)
