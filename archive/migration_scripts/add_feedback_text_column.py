"""Add FeedbackText column to APP_IncidentCaseSatisfaction table"""
import sys
sys.path.insert(0, 'backend')

from core.database import get_connection

conn = get_connection()
cursor = conn.cursor()

# First check if it has the column
cursor.execute("""
SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'APP_IncidentCaseSatisfaction' AND COLUMN_NAME = 'FeedbackText'
""")
exists = cursor.fetchone()

if exists:
    print('FeedbackText column already exists')
else:
    # Add the column if it does not exist - need to use IF NOT EXISTS pattern
    cursor.execute("""
    IF NOT EXISTS (
        SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS 
        WHERE TABLE_NAME = 'APP_IncidentCaseSatisfaction' AND COLUMN_NAME = 'FeedbackText'
    )
    BEGIN
        ALTER TABLE dbo.APP_IncidentCaseSatisfaction
        ADD FeedbackText NVARCHAR(1000) NULL
    END
    """)
    conn.commit()
    print('Added FeedbackText column successfully!')

cursor.close()
conn.close()
