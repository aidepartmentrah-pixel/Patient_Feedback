/*
==============================================================================
RESERVE DOCTOR TABLE - Phase 1 (CORRECTED - IDENTICAL TO HOSPITAL TABLE)
==============================================================================
Purpose: Create a reserve table IDENTICAL to APP_LOOKUP_DOCTOR structure.
         No extra fields - perfect mirror of hospital table.
         
Author: Phase 1 - Add New Doctor Feature (Corrected)
Date: 2026-01-20

Note: This table is IDENTICAL to APP_LOOKUP_DOCTOR to ensure perfect 
      compatibility when merging results.
==============================================================================
*/

-- Drop old reserve table if it exists
IF OBJECT_ID('dbo.APP_RESERVE_DOCTOR', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing APP_RESERVE_DOCTOR table...'
    DROP TABLE dbo.APP_RESERVE_DOCTOR
END
GO

-- Create the reserve doctor table (IDENTICAL to APP_LOOKUP_DOCTOR)
PRINT 'Creating APP_RESERVE_DOCTOR table (identical to hospital table)...'
GO

CREATE TABLE dbo.APP_RESERVE_DOCTOR (
    -- Primary Key
    DoctorID INT IDENTITY(1,1) PRIMARY KEY,
    
    -- Doctor Information (matching hospital table exactly)
    DoctorName NVARCHAR(200) NOT NULL,
    Specialty NVARCHAR(200) NULL,
    IsActive BIT NOT NULL DEFAULT 1,
    SourceSystem NVARCHAR(100) NULL DEFAULT 'MANUAL',
    LastSyncedAt DATETIME NULL DEFAULT GETDATE()
)
GO

-- Create indexes for faster searches (same as hospital table pattern)
CREATE INDEX IX_ReserveDoctor_DoctorName ON dbo.APP_RESERVE_DOCTOR(DoctorName)
GO

CREATE INDEX IX_ReserveDoctor_IsActive ON dbo.APP_RESERVE_DOCTOR(IsActive)
GO

PRINT 'APP_RESERVE_DOCTOR table created successfully!'
GO

-- Insert test data for verification
PRINT 'Inserting test data...'
GO

INSERT INTO dbo.APP_RESERVE_DOCTOR 
    (DoctorName, Specialty, IsActive, SourceSystem, LastSyncedAt)
VALUES 
    ('Dr. Ahmed Al-Rashid', 'Interventional Cardiology', 1, 'MANUAL', GETDATE()),
    ('Dr. Fatima Hassan', 'Neonatology', 1, 'MANUAL', GETDATE()),
    ('Dr. Mohammed Ibrahim', 'Emergency Physician', 1, 'MANUAL', GETDATE())
GO

PRINT 'Test data inserted successfully!'
GO

-- Verification query
PRINT 'Verifying table creation and data...'
GO

SELECT 
    DoctorID,
    DoctorName,
    Specialty,
    CASE WHEN IsActive = 1 THEN 'Active' ELSE 'Inactive' END AS Status,
    SourceSystem,
    LastSyncedAt
FROM dbo.APP_RESERVE_DOCTOR
ORDER BY DoctorID
GO

PRINT '============================================'
PRINT 'Phase 1 Complete - Reserve table is ready!'
PRINT 'Table structure is IDENTICAL to hospital table'
PRINT '============================================'
GO
