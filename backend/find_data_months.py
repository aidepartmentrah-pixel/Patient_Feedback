"""
Quick script to find months with complaint data
"""
import sys
sys.path.insert(0, r"C:\Users\IT\Documents\GitHub Repository\Patient_Feedback\backend")

from api.db_layer.reports_db import get_connection

conn = get_connection()
cursor = conn.cursor()

query = """
SELECT 
    YEAR(FeedbackRecievedDate) as year,
    MONTH(FeedbackRecievedDate) as month,
    COUNT(*) as count
FROM dbo.APP_IncidentCase
GROUP BY YEAR(FeedbackRecievedDate), MONTH(FeedbackRecievedDate)
ORDER BY year DESC, month DESC
"""

cursor.execute(query)
rows = cursor.fetchall()

print("Available months with data:")
print("=" * 40)
for row in rows[:20]:
    print(f"{row.year}-{row.month:02d}: {row.count} complaints")

conn.close()
