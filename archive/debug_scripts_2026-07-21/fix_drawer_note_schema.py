"""
Fix APP_DrawerNote table schema - add PatientAdmissionID column
"""
from core.database import get_connection

def main():
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if PatientAdmissionID column exists
        cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'APP_DrawerNote' AND COLUMN_NAME = 'PatientAdmissionID'
        """)
        exists = cursor.fetchone()
        print(f"PatientAdmissionID column exists: {exists is not None}")
        
        if not exists:
            # Add the column
            print("Adding PatientAdmissionID column to APP_DrawerNote...")
            cursor.execute("""
                ALTER TABLE dbo.APP_DrawerNote 
                ADD PatientAdmissionID INT NULL
            """)
            conn.commit()
            print("Column added successfully!")
        
        # Check VW_PatientAdmission view columns
        cursor.execute("""
            SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = 'VW_PatientAdmission'
        """)
        cols = [row[0] for row in cursor.fetchall()]
        print(f"VW_PatientAdmission columns: {cols[:15]}")
        
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    main()
