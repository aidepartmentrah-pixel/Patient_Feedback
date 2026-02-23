"""Test that database queries work with renamed tables."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_connection
from core.table_config import HR_EMPLOYEES_TABLE, PATIENT_ADMISSION_TABLE, DOCTORS_TABLE

def test_queries():
    conn = get_connection()
    cursor = conn.cursor()

    print('Testing queries with new table names...')

    # Test HR Employees
    cursor.execute(f'SELECT TOP 1 * FROM {HR_EMPLOYEES_TABLE}')
    row = cursor.fetchone()
    print(f'  HR_EMPLOYEES ({HR_EMPLOYEES_TABLE}): {"OK" if row else "EMPTY"}')

    # Test Patient Admission
    cursor.execute(f'SELECT TOP 1 * FROM {PATIENT_ADMISSION_TABLE}')
    row = cursor.fetchone()
    print(f'  PATIENT_ADMISSION ({PATIENT_ADMISSION_TABLE}): {"OK" if row else "EMPTY"}')

    # Test Doctors
    cursor.execute(f'SELECT TOP 1 * FROM {DOCTORS_TABLE}')
    row = cursor.fetchone()
    print(f'  DOCTORS ({DOCTORS_TABLE}): {"OK" if row else "EMPTY"}')

    cursor.close()
    conn.close()
    print('All queries successful!')

if __name__ == '__main__':
    test_queries()
