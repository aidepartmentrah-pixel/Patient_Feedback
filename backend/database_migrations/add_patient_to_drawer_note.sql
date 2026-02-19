-- ============================================================
-- Migration: Add Patient Foreign Key to APP_DrawerNote
-- Date: 2026-02-19
-- Purpose: Link drawer notes to patients from patient admission tables
-- ============================================================

-- Step 1: Add PatientAdmissionID column (nullable to preserve existing notes)
IF NOT EXISTS (
    SELECT 1 
    FROM sys.columns 
    WHERE object_id = OBJECT_ID('dbo.APP_DrawerNote') 
    AND name = 'PatientAdmissionID'
)
BEGIN
    ALTER TABLE dbo.APP_DrawerNote
    ADD PatientAdmissionID int NULL;
    
    PRINT 'Added PatientAdmissionID column to APP_DrawerNote';
END
ELSE
BEGIN
    PRINT 'PatientAdmissionID column already exists in APP_DrawerNote';
END
GO

-- Step 2: Add index on PatientAdmissionID for efficient filtering
IF NOT EXISTS (
    SELECT 1 
    FROM sys.indexes 
    WHERE object_id = OBJECT_ID('dbo.APP_DrawerNote') 
    AND name = 'IX_APP_DrawerNote_PatientAdmissionID'
)
BEGIN
    CREATE NONCLUSTERED INDEX IX_APP_DrawerNote_PatientAdmissionID
    ON dbo.APP_DrawerNote (PatientAdmissionID)
    WHERE PatientAdmissionID IS NOT NULL;
    
    PRINT 'Created index IX_APP_DrawerNote_PatientAdmissionID';
END
ELSE
BEGIN
    PRINT 'Index IX_APP_DrawerNote_PatientAdmissionID already exists';
END
GO

-- Note: Foreign key constraint is NOT added because patients can come from:
-- 1. APP_VIEWTABLE_PATIENT_ADMISSION (view table - synced from external source)
-- 2. APP_RESERVE_PATIENT (manual fallback table)
-- The application layer handles patient validation.

-- ============================================================
-- Verification Query (run after migration)
-- ============================================================
/*
SELECT 
    c.name AS ColumnName,
    t.name AS DataType,
    c.is_nullable AS IsNullable
FROM sys.columns c
JOIN sys.types t ON c.user_type_id = t.user_type_id
WHERE object_id = OBJECT_ID('dbo.APP_DrawerNote')
ORDER BY c.column_id;
*/
