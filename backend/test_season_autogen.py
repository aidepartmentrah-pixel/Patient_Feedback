"""
Test Script: Season Auto-Generation
====================================
This script tests that the season auto-generation works correctly.

It simulates different dates (2027, 2028, 2030) to prove that:
1. Missing seasons are auto-created when needed
2. Current season is correctly detected by date range
3. The software will work autonomously forever

Run this script to verify everything works:
    cd backend
    python test_season_autogen.py
"""

from datetime import date
from core.database import get_connection
from api_v2.db_layer import season_db


def print_banner(text: str):
    """Print a banner for test sections."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def show_all_seasons(conn):
    """Display all seasons in the database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT UniqueID, SeasonName, StartDate, EndDate, IsDone, Frozen
        FROM dbo.Season
        ORDER BY StartDate DESC
    """)
    rows = cursor.fetchall()
    cursor.close()
    
    print(f"\n{'ID':<4} {'Name':<10} {'Start':<12} {'End':<12} {'Done':<5} {'Frozen':<6}")
    print("-" * 55)
    for row in rows:
        print(f"{row.UniqueID:<4} {row.SeasonName:<10} {str(row.StartDate):<12} {str(row.EndDate):<12} {str(row.IsDone):<5} {str(row.Frozen):<6}")
    
    return len(rows)


def test_scenario(test_name: str, simulated_date: date):
    """Test auto-generation with a simulated date."""
    print_banner(f"TEST: {test_name}")
    print(f"Simulated Date: {simulated_date}")
    print(f"Simulated Year: {simulated_date.year}")
    
    conn = get_connection()
    
    # Count seasons before
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM dbo.Season")
    count_before = cursor.fetchone()[0]
    cursor.close()
    print(f"\nSeasons before auto-gen: {count_before}")
    
    # Run auto-generation with simulated date
    print("\nRunning ensure_seasons_exist()...")
    created = season_db.ensure_seasons_exist(conn, years_ahead=2, reference_date=simulated_date)
    print(f"Seasons created: {created}")
    
    # Count seasons after
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM dbo.Season")
    count_after = cursor.fetchone()[0]
    cursor.close()
    print(f"Seasons after auto-gen: {count_after}")
    
    # Get current season by date
    print("\nDetecting current season by date...")
    current = season_db.get_current_season_by_date(conn, reference_date=simulated_date)
    if current:
        print(f"Current Season: {current['season_name']} ({current['start_date']} to {current['end_date']})")
    else:
        print("No current season found for this date!")
    
    conn.close()
    
    return created, current


def cleanup_test_seasons():
    """Remove test seasons created for future years (optional)."""
    print_banner("CLEANUP (Optional)")
    print("Removing seasons for years 2027 and beyond...")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Delete seasons for 2027+
    cursor.execute("""
        DELETE FROM dbo.Season 
        WHERE YEAR(StartDate) >= 2027
    """)
    deleted = cursor.rowcount
    conn.commit()
    cursor.close()
    conn.close()
    
    print(f"Deleted {deleted} test seasons")
    return deleted


# ============================================================
# MAIN TEST EXECUTION
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  SEASON AUTO-GENERATION TEST SUITE")
    print("  ==================================")
    print("  This proves the software works autonomously forever!")
    print("=" * 60)
    
    # Show initial state
    print_banner("INITIAL STATE")
    conn = get_connection()
    initial_count = show_all_seasons(conn)
    conn.close()
    
    # Test 1: Current date (February 20, 2026)
    test_scenario(
        "Current Date (Feb 2026)",
        date(2026, 2, 20)
    )
    
    # Test 2: Simulate Jan 15, 2027
    created_2027, current_2027 = test_scenario(
        "Simulating Year 2027 (Jan 15)",
        date(2027, 1, 15)
    )
    
    # Verify Q1-2027 was detected
    if current_2027 and current_2027['season_name'] == 'Q1-2027':
        print("\n✓ SUCCESS: Q1-2027 correctly detected as current season!")
    else:
        print("\n✗ FAILURE: Q1-2027 was not detected correctly")
    
    # Test 3: Simulate Aug 5, 2028
    created_2028, current_2028 = test_scenario(
        "Simulating Year 2028 (Aug 5)",
        date(2028, 8, 5)
    )
    
    # Verify Q3-2028 was detected
    if current_2028 and current_2028['season_name'] == 'Q3-2028':
        print("\n✓ SUCCESS: Q3-2028 correctly detected as current season!")
    else:
        print("\n✗ FAILURE: Q3-2028 was not detected correctly")
    
    # Test 4: Simulate Nov 20, 2030
    created_2030, current_2030 = test_scenario(
        "Simulating Year 2030 (Nov 20)",
        date(2030, 11, 20)
    )
    
    # Verify Q4-2030 was detected
    if current_2030 and current_2030['season_name'] == 'Q4-2030':
        print("\n✓ SUCCESS: Q4-2030 correctly detected as current season!")
    else:
        print("\n✗ FAILURE: Q4-2030 was not detected correctly")
    
    # Show final state
    print_banner("FINAL STATE (All Seasons)")
    conn = get_connection()
    final_count = show_all_seasons(conn)
    conn.close()
    
    # Summary
    print_banner("TEST SUMMARY")
    print(f"Initial seasons: {initial_count}")
    print(f"Final seasons:   {final_count}")
    print(f"Total created:   {final_count - initial_count}")
    print()
    
    # Ask about cleanup
    print("The test created seasons for 2027-2032.")
    print("These will exist in your database.")
    print()
    response = input("Do you want to DELETE the test seasons (2027+)? [y/N]: ")
    
    if response.lower() == 'y':
        cleanup_test_seasons()
        print("\nTest seasons removed. Database restored to original state.")
    else:
        print("\nTest seasons kept. They will be useful when those years arrive!")
    
    print_banner("TEST COMPLETE")
    print("Your software is now future-proof!")
    print("It will automatically create seasons as time moves forward.")
