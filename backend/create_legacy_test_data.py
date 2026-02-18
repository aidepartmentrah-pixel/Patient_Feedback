"""
Create test legacy data for K-SVC-1 testing

This script populates the legacy IncidentRequestCase table with test data
that can be used for migration testing.
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


def create_legacy_test_data():
    """Insert test legacy cases"""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 80)
    print("CREATING LEGACY TEST DATA")
    print("=" * 80)
    
    try:
        # Get existing IncidentRequest IDs
        cursor.execute("SELECT TOP 10 UniqueID FROM IncidentRequest ORDER BY DateAndTimeRecieved DESC")
        request_ids = [row[0] for row in cursor.fetchall()]
        
        if not request_ids:
            print("❌ No IncidentRequest records found. Cannot create test cases.")
            return False
        
        print(f"\n✅ Found {len(request_ids)} IncidentRequest records")
        print(f"   Will create legacy cases for these requests")
        
        # Create legacy cases for each request
        created_count = 0
        for i, request_id in enumerate(request_ids, 1):
            try:
                # Create description with varying lengths for testing
                description = f"Legacy case description {i} - This is a test legacy case that needs to be migrated to the new system. It contains historical data that should be preserved during migration. Additional details may include patient concerns, initial assessments, and preliminary findings from the legacy system."
                
                # For some cases, add extra long description to test truncation
                if i % 3 == 0:
                    description += " " + ("Extended content. " * 50)  # Make it very long
                
                cursor.execute("""
                    INSERT INTO IncidentRequestCase (
                        IncidentRequestID,
                        Description,
                        Note,
                        DateAndTimeCreated,
                        Frozen
                    ) VALUES (?, ?, ?, GETDATE(), 0)
                """, 
                    request_id,
                    description,
                    f"Legacy note for case {i} - additional information"
                )
                created_count += 1
                print(f"  ✅ Created legacy case for request {request_id}")
                
            except Exception as e:
                print(f"  ⚠️  Skipped request {request_id}: {e}")
        
        conn.commit()
        
        # Verify creation
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCase")
        total_cases = cursor.fetchone()[0]
        
        print(f"\n{'=' * 80}")
        print(f"✅ Created {created_count} legacy test cases")
        print(f"   Total IncidentRequestCase rows: {total_cases}")
        print('=' * 80)
        
        # Show sample of created data
        cursor.execute("""
            SELECT TOP 3
                irc.UniqueID,
                ir.PatientName,
                LEFT(irc.Description, 50) as Preview
            FROM IncidentRequestCase irc
            INNER JOIN IncidentRequest ir ON ir.UniqueID = irc.IncidentRequestID
        """)
        
        print("\n📋 Sample created records:")
        for row in cursor.fetchall():
            print(f"  Case {row[0]}: {row[1]} - {row[2]}...")
        
        return created_count > 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
        return False
    
    finally:
        cursor.close()
        conn.close()


if __name__ == "__main__":
    success = create_legacy_test_data()
    sys.exit(0 if success else 1)
