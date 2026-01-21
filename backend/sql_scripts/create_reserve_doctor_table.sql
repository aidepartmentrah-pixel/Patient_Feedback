/*
==============================================================================
RESERVE DOCTOR TABLE - Phase 1
==============================================================================
Purpose: Create a reserve table for user-added doctors separate from the 
         hospital's view (APP_VIEWTABLE_VW_DOCTORS or APP_LOOKUP_DOCTOR).
         
Author: Phase 1 - Add New Doctor Feature
Date: 2026-01-20

Note: This table allows the application to add doctors without needing 
      write access to the hospital's main database.
==============================================================================
*/

-- Drop table if it exists (for clean re-runs in development)
IF OBJECT_ID('dbo.APP_RESERVE_DOCTOR', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing APP_RESERVE_DOCTOR table...'
    DROP TABLE dbo.APP_RESERVE_DOCTOR
END
GO

-- Create the reserve doctor table
PRINT 'Creating APP_RESERVE_DOCTOR table...'
GO

CREATE TABLE dbo.APP_RESERVE_DOCTOR (
    -- Primary Key
    DoctorID INT IDENTITY(1,1) PRIMARY KEY,
    
    -- Required Fields
    EmployeeID NVARCHAR(50) NOT NULL,
    DoctorName NVARCHAR(200) NOT NULL,
    
    -- Optional Fields (matching hospital table structure)
    Department NVARCHAR(200) NULL,
    Specialty NVARCHAR(200) NULL,
    Email NVARCHAR(200) NULL,
    Phone NVARCHAR(50) NULL,
    LicenseNumber NVARCHAR(100) NULL,
    HireDate DATE NULL DEFAULT GETDATE(),
    
    -- Status Field (1=Active, 0=Inactive, 2=Suspended)
    IsActive INT NOT NULL DEFAULT 1,
    
    -- Audit Fields
    CreatedAt DATETIME NOT NULL DEFAULT GETDATE(),
    UpdatedAt DATETIME NOT NULL DEFAULT GETDATE(),
    
    -- Unique constraint on EmployeeID
    CONSTRAINT UQ_ReserveDoctor_EmployeeID UNIQUE (EmployeeID)
)
GO

-- Create index for faster searches
CREATE INDEX IX_ReserveDoctor_DoctorName ON dbo.APP_RESERVE_DOCTOR(DoctorName)
GO

CREATE INDEX IX_ReserveDoctor_Department ON dbo.APP_RESERVE_DOCTOR(Department)
GO

CREATE INDEX IX_ReserveDoctor_IsActive ON dbo.APP_RESERVE_DOCTOR(IsActive)
GO

PRINT 'APP_RESERVE_DOCTOR table created successfully!'
GO

-- Insert test data for verification
PRINT 'Inserting test data...'
GO

INSERT INTO dbo.APP_RESERVE_DOCTOR 
    (EmployeeID, DoctorName, Department, Specialty, Email, Phone, LicenseNumber, HireDate, IsActive)
VALUES 
    ('TEST-DOC-001', 'Dr. Ahmed Al-Rashid', 'Cardiology', 'Interventional Cardiology', 'ahmed.alrashid@hospital.com', '+966501234567', 'LIC-CARD-001', '2024-01-15', 1),
    ('TEST-DOC-002', 'Dr. Fatima Hassan', 'Pediatrics', 'Neonatology', 'fatima.hassan@hospital.com', '+966509876543', 'LIC-PED-002', '2023-06-20', 1),
    ('TEST-DOC-003', 'Dr. Mohammed Ibrahim', 'Emergency Medicine', 'Emergency Physician', 'mohammed.ibrahim@hospital.com', '+966505555555', 'LIC-EM-003', '2025-03-10', 1)
GO

PRINT 'Test data inserted successfully!'
GO

-- Verification query
PRINT 'Verifying table creation and data...'
GO

SELECT 
    DoctorID,
    EmployeeID,
    DoctorName,
    Department,
    Specialty,
    Email,
    Phone,
    LicenseNumber,
    HireDate,
    CASE 
        WHEN IsActive = 1 THEN 'Active'
        WHEN IsActive = 2 THEN 'Suspended'
        ELSE 'Inactive'
    END AS Status,
    CreatedAt,
    UpdatedAt
FROM dbo.APP_RESERVE_DOCTOR
ORDER BY DoctorID
GO

PRINT '============================================'
PRINT 'Phase 1 Complete - Reserve table is ready!'
PRINT '============================================'
GO
