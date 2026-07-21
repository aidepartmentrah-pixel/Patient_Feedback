"""
Check legacy tables for data availability
"""

import sys
from pathlib import Path

backend_path = Path(__file__).parent
sys.path.insert(0, str(backend_path))

from core.database import get_connection


def check_legacy_tables():
    """Check what legacy data exists"""
    conn = get_connection()
    cursor = conn.cursor()
    
    print("=" * 80)
    print("LEGACY TABLES DATA CHECK")
    print("=" * 80)
    
    # Check IncidentRequest
    print("\n📋 IncidentRequest:")
    cursor.execute("SELECT COUNT(*) FROM IncidentRequest")
    request_count = cursor.fetchone()[0]
    print(f"  Total rows: {request_count}")
    
    if request_count > 0:
        cursor.execute("""
            SELECT TOP 5 
                UniqueID, 
                PatientName, 
                CONVERT(VARCHAR(19), DateAndTimeRecieved, 121) as ReceivedDate
            FROM IncidentRequest
            ORDER BY DateAndTimeRecieved DESC
        """)
        print("\n  Sample rows:")
        for row in cursor.fetchall():
            print(f"    ID: {row[0]}, Patient: {row[1]}, Date: {row[2]}")
    
    # Check IncidentRequestCase
    print("\n📋 IncidentRequestCase:")
    cursor.execute("SELECT COUNT(*) FROM IncidentRequestCase")
    case_count = cursor.fetchone()[0]
    print(f"  Total rows: {case_count}")
    
    if case_count > 0:
        cursor.execute("""
            SELECT TOP 5 
                UniqueID, 
                IncidentRequestID, 
                LEFT(Description, 50) as Description
            FROM IncidentRequestCase
        """)
        print("\n  Sample rows:")
        for row in cursor.fetchall():
            print(f"    Case ID: {row[0]}, Request ID: {row[1]}, Desc: {row[2]}...")
    
    # Check if the join would work
    print("\n🔗 JOIN CHECK:")
    cursor.execute("""
        SELECT COUNT(*)
        FROM IncidentRequestCase irc
        INNER JOIN IncidentRequest ir ON ir.UniqueID = irc.IncidentRequestID
    """)
    joined_count = cursor.fetchone()[0]
    print(f"  Joinable cases: {joined_count}")
    
    # Check IncidentRequestCaseAction
    print("\n📋 IncidentRequestCaseAction:")
    cursor.execute("SELECT COUNT(*) FROM IncidentRequestCaseAction")
    action_count = cursor.fetchone()[0]
    print(f"  Total rows: {action_count}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    if case_count == 0:
        print("❌ No legacy cases exist - need to create test data")
        print("   IncidentRequestCase table is empty")
    elif joined_count == 0:
        print("⚠️  Cases exist but don't join with requests")
        print("   Data integrity issue")
    else:
        print(f"✅ {joined_count} legacy cases available for migration")
    print("=" * 80)
    
    cursor.close()
    conn.close()


if __name__ == "__main__":
    check_legacy_tables()
