"""
Rename external system tables to match production view names.
Run this once to update the database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_connection

def rename_tables():
    conn = get_connection()
    cursor = conn.cursor()

    # Rename tables to match production view names
    renames = [
        ('APP_VIEWTABLE_HR_EMPLOYEES', 'VW_HrEmployeeProfileView'),
        ('APP_VIEWTABLE_PATIENT_ADMISSION', 'VW_PatientAdmission'),
        ('APP_VIEWTABLE_VW_DOCTORS', 'VW_Doctors')
    ]

    print('Renaming tables to match production view names...')
    for old_name, new_name in renames:
        try:
            sql = f"EXEC sp_rename '{old_name}', '{new_name}'"
            cursor.execute(sql)
            print(f'  OK: {old_name} -> {new_name}')
        except Exception as e:
            print(f'  ERROR {old_name}: {str(e)}')

    conn.commit()
    print()
    print('Done! Verifying...')

    cursor.execute("""
        SELECT TABLE_NAME, TABLE_TYPE 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_NAME LIKE 'VW_%'
        ORDER BY TABLE_NAME
    """)
    print('Tables/Views with VW_ prefix:')
    for row in cursor.fetchall():
        print(f'  {row.TABLE_NAME} ({row.TABLE_TYPE})')

    cursor.close()
    conn.close()

if __name__ == '__main__':
    rename_tables()
