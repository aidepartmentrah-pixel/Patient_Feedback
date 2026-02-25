/*
============================================================
ADD IsActive COLUMN TO APP_RESERVE_PATIENT TABLE
============================================================

Purpose: Enable soft-delete functionality for reserve patients.
When IsActive = 0, the patient is considered "deleted" but data is preserved.

This matches the pattern used for APP_RESERVE_DOCTOR soft-delete.

Execution: Run this script once to add the column.
============================================================
*/

-- Check if column already exists
IF NOT EXISTS (
    SELECT 1 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_RESERVE_PATIENT' 
    AND COLUMN_NAME = 'IsActive'
)
BEGIN
    PRINT 'Adding IsActive column to APP_RESERVE_PATIENT...'
    
    -- Add IsActive column with default value 1 (active)
    ALTER TABLE dbo.APP_RESERVE_PATIENT
    ADD IsActive BIT NOT NULL DEFAULT 1;
    
    PRINT 'IsActive column added successfully!'
    PRINT 'All existing patients are set to IsActive = 1 (active)'
END
ELSE
BEGIN
    PRINT 'IsActive column already exists in APP_RESERVE_PATIENT - no action needed'
END
GO

-- Verify the column was added
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    COLUMN_DEFAULT
FROM INFORMATION_SCHEMA.COLUMNS 
WHERE TABLE_NAME = 'APP_RESERVE_PATIENT' 
AND COLUMN_NAME = 'IsActive';

PRINT 'Migration complete!'
