"""
Test direct INSERT into AdminsrationUnit
"""
import sys
from pathlib import Path

backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

try:
    print("\n=== Testing INSERT into AdminsrationUnit ===")
    
    # Try to insert a test section
    insert_query = """
        INSERT INTO dbo.AdminsrationUnit (Name, ParentID, Type, Frozen, CreateDate)
        VALUES (?, ?, 324, 0, GETDATE())
    """
    
    test_name = "TEST SECTION FROM PYTHON"
    parent_id = 5  # Valid department we found
    
    print(f"Inserting: Name='{test_name}', ParentID={parent_id}, Type=324")
    
    cursor.execute(insert_query, (test_name, parent_id))
    
    # Get the new ID
    cursor.execute("SELECT CAST(SCOPE_IDENTITY() AS INT) AS section_id")
    result = cursor.fetchone()
    
    if result and result.section_id:
        new_id = result.section_id
        print(f"✓ INSERT successful! New UniqueID: {new_id}")
        
        # Verify it was created
        cursor.execute("SELECT * FROM dbo.AdminsrationUnit WHERE UniqueID = ?", (new_id,))
        verify = cursor.fetchone()
        if verify:
            print(f"✓ Verified: {verify.Name}")
        
        # Rollback (don't actually keep it)
        conn.rollback()
        print("✓ Rolled back (test only)")
    else:
        print("✗ SCOPE_IDENTITY() returned NULL")
        conn.rollback()
        
except Exception as e:
    print(f"✗ Error: {str(e)}")
    conn.rollback()

finally:
    cursor.close()
    conn.close()
