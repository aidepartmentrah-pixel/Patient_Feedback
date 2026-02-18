"""
Create test action data for K-SVC-2 testing

This script populates the IncidentRequestCaseAction table with test data
to enable complete action ordering validation.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


def create_action_test_data():
    """Insert test actions for legacy cases"""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 80)
    print("CREATING LEGACY ACTION TEST DATA")
    print("=" * 80)
    
    try:
        # Get existing case IDs
        cursor.execute("SELECT TOP 3 UniqueID FROM IncidentRequestCase ORDER BY UniqueID")
        case_ids = [row[0] for row in cursor.fetchall()]
        
        if not case_ids:
            print("❌ No IncidentRequestCase records found.")
            return False
        
        print(f"\n✅ Found {len(case_ids)} cases to add actions for")
        
        created_count = 0
        
        # Create multiple actions for first case (to test ordering)
        if len(case_ids) >= 1:
            case_id = case_ids[0]
            base_time = datetime(2025, 11, 27, 10, 0, 0)
            
            actions = [
                ("First action taken", "Initial response note", base_time),
                ("Second follow-up action", "Follow-up details", base_time + timedelta(hours=2)),
                ("Third corrective action", "Corrective measures", base_time + timedelta(hours=5)),
                ("Final closure action", "Case closure notes", base_time + timedelta(days=1))
            ]
            
            print(f"\n📋 Creating {len(actions)} actions for case {case_id}:")
            for desc, note, action_time in actions:
                try:
                    cursor.execute("""
                        INSERT INTO IncidentRequestCaseAction (
                            IncidentRequestCaseID,
                            Description,
                            Note,
                            DateAndTimeCreated
                        ) VALUES (?, ?, ?, ?)
                    """, case_id, desc, note, action_time)
                    created_count += 1
                    print(f"  ✅ {action_time.strftime('%Y-%m-%d %H:%M')} - {desc}")
                except Exception as e:
                    print(f"  ⚠️  Failed: {e}")
        
        # Create single action for second case
        if len(case_ids) >= 2:
            case_id = case_ids[1]
            try:
                cursor.execute("""
                    INSERT INTO IncidentRequestCaseAction (
                        IncidentRequestCaseID,
                        Description,
                        Note,
                        DateAndTimeCreated
                    ) VALUES (?, ?, ?, ?)
                """, case_id, "Single action for case", "This case has only one action", datetime.now())
                created_count += 1
                print(f"\n📋 Created 1 action for case {case_id}")
            except Exception as e:
                print(f"  ⚠️  Failed for case {case_id}: {e}")
        
        # Leave third case without actions (for empty actions test)
        if len(case_ids) >= 3:
            print(f"\n📋 Case {case_ids[2]} left without actions (for empty test)")
        
        conn.commit()
        
        # Verify creation
        cursor.execute("SELECT COUNT(*) FROM IncidentRequestCaseAction")
        total_actions = cursor.fetchone()[0]
        
        print(f"\n{'=' * 80}")
        print(f"✅ Created {created_count} action records")
        print(f"   Total IncidentRequestCaseAction rows: {total_actions}")
        print('=' * 80)
        
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
    success = create_action_test_data()
    sys.exit(0 if success else 1)
