"""Direct test of the trends query to see the full SQL error"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from core.database import get_connection
from datetime import datetime, timedelta

# Test the FIRST query (base query) from red_flags_service
from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
to_date = datetime.now().strftime("%Y-%m-%d")

date_format = "FORMAT(c.FeedbackRecievedDate, 'MMM yyyy')"
date_group = "YEAR(c.FeedbackRecievedDate), MONTH(c.FeedbackRecievedDate)"

base_query = f"""
    SELECT 
        {date_format} as period,
        COUNT(*) as count
    FROM dbo.APP_IncidentCase c
    WHERE c.ClinicalRiskTypeID = 2
    AND c.FeedbackRecievedDate >= ?
    AND c.FeedbackRecievedDate <= ?
    GROUP BY {date_group}
    ORDER BY {date_group}
"""

print("=" * 80)
print("Testing Base Query (FIRST query in trends function)")
print("=" * 80)
print("\nQuery:")
print(base_query)
print("\nParameters:")
print(f"  from_date: {from_date}")
print(f"  to_date: {to_date}")
print()

try:
    conn = get_connection()
    cursor = conn.cursor()
    
    print("Executing query...")
    cursor.execute(base_query, [from_date, to_date])
    
    print("✅ Query executed successfully!")
    print("\nResults:")
    
    rows = cursor.fetchall()
    print(f"  Found {len(rows)} periods")
    
    for row in rows[:5]:  # Show first 5
        print(f"  - {row[0]}: Count={row[1]}")
    
    cursor.close()
    conn.close()
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
