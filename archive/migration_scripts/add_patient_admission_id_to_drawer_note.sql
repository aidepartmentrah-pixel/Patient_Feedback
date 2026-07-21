-- Migration: Add PatientAdmissionID column to APP_DrawerNote table
-- Date: 2026-02-25
-- Purpose: Enable linking drawer notes to patients (optional relationship)

-- Check if column exists before adding
IF NOT EXISTS (
    SELECT 1 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_NAME = 'APP_DrawerNote' 
    AND COLUMN_NAME = 'PatientAdmissionID'
    AND TABLE_SCHEMA = 'dbo'
)
BEGIN
    -- Add the PatientAdmissionID column (nullable - patients are optional)
    ALTER TABLE dbo.APP_DrawerNote
    ADD PatientAdmissionID INT NULL;
    
    PRINT 'Added PatientAdmissionID column to APP_DrawerNote table';
END
ELSE
BEGIN
    PRINT 'PatientAdmissionID column already exists in APP_DrawerNote table';
END
GO
