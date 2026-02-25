"""
Add UNIVERSAL_SECTION role to APP_Roles table.
Run once to create the operational bridge role.
"""
from backend.core.database import get_connection

def add_universal_section_role():
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        # Check if role already exists
        cursor.execute("SELECT COUNT(*) FROM dbo.APP_Roles WHERE RoleCode = 'UNIVERSAL_SECTION'")
        if cursor.fetchone()[0] == 0:
            cursor.execute("""
                INSERT INTO dbo.APP_Roles (RoleCode, RoleNameEn, RoleNameAr)
                VALUES ('UNIVERSAL_SECTION', 'Universal Section Administrator', N'مسؤول الأقسام الشامل')
            """)
            conn.commit()
            print('✅ Role UNIVERSAL_SECTION added to database')
        else:
            print('ℹ️ Role UNIVERSAL_SECTION already exists')
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    add_universal_section_role()
