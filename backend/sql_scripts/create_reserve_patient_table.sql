/*
====================================================================
CREATE RESERVE PATIENT TABLE
====================================================================
Purpose: Create APP_RESERVE_PATIENT table with IDENTICAL structure 
         to APP_VIEWTABLE_PATIENT_ADMISSION (hospital table)

Strategy: Dual-source pattern
  - Hospital table: APP_VIEWTABLE_PATIENT_ADMISSION (read-only)
  - Reserve table:  APP_RESERVE_PATIENT (writable)
  - Merge both sources using UNION queries

Author: System
Date: 2026-01-20
====================================================================
*/

-- Drop table if exists (for clean re-runs)
IF OBJECT_ID('dbo.APP_RESERVE_PATIENT', 'U') IS NOT NULL
BEGIN
    PRINT 'Dropping existing APP_RESERVE_PATIENT table...'
    DROP TABLE dbo.APP_RESERVE_PATIENT
END
GO

-- Create reserve patient table with IDENTICAL structure
CREATE TABLE dbo.APP_RESERVE_PATIENT (
    -- Primary Key
    PatientAdmissionID INT IDENTITY(100000,1) PRIMARY KEY,  -- Start at 100000 to avoid collision with hospital IDs
    
    -- Admission Information
    CaseID INT NULL,
    AdmissionDate DATETIME NULL,
    AdmissionTime DATETIME NULL,
    DischargeDate DATETIME NULL,
    
    -- Patient Name Information
    FullName NVARCHAR(250) NULL,
    FirstName NVARCHAR(150) NULL,
    MiddleName NVARCHAR(150) NULL,
    LastName NVARCHAR(150) NULL,
    Spouse NVARCHAR(150) NULL,
    MotherName NVARCHAR(150) NULL,
    
    -- Contact Information
    AddressLine1 NVARCHAR(300) NULL,
    AddressLine2 NVARCHAR(300) NULL,
    PhoneNumber1 NVARCHAR(50) NULL,
    PhoneNumber2 NVARCHAR(50) NULL,
    
    -- Personal Information
    SEX NVARCHAR(10) NULL,
    BirthDate DATE NULL,
    DocumentNumber NVARCHAR(100) NULL,
    
    -- Discharge & Reference
    DischargeNumber NVARCHAR(100) NULL,
    Reference NVARCHAR(200) NULL,
    Diagnoses NVARCHAR(MAX) NULL,
    
    -- Doctor & Service Information
    DoctorID INT NULL,
    AssistantDoctorID INT NULL,
    MedicalServiceID INT NULL,
    MedicalServiceClassID INT NULL,
    
    -- Room & Bed Information
    RoomNumber NVARCHAR(50) NULL,
    BED NVARCHAR(50) NULL,
    Stay INT NULL,
    
    -- Serial & File Numbers
    SerialNumber NVARCHAR(100) NULL,
    SerialSequence NVARCHAR(100) NULL,
    MedicalFileNumber NVARCHAR(100) NULL,
    
    -- Status & Demographics
    MaritalStatusID INT NULL,
    EmployeeID INT NULL,
    NationalityID INT NULL,
    
    -- Location Information
    LocationTypeID INT NULL,
    RoomBedID INT NULL,
    CareCenterID INT NULL,
    
    -- Guarantor Information (1st)
    GuarantorCode NVARCHAR(100) NULL,
    GuarantorDocumentNumber NVARCHAR(100) NULL,
    GuarantorClassNumber NVARCHAR(100) NULL,
    
    -- Guarantor Information (2nd)
    SecondGuarantorCode NVARCHAR(100) NULL,
    SecondGuarantorDocumentNumber NVARCHAR(100) NULL,
    SecondGuarantorClassNumber NVARCHAR(100) NULL,
    
    -- Guarantor Information (3rd)
    ThirdGuarantorCode NVARCHAR(100) NULL,
    ThirdGuarantorDocumentNumber NVARCHAR(100) NULL,
    ThirdGuarantorClassNumber NVARCHAR(100) NULL,
    
    -- Guarantor Amounts
    GuarantorAmount DECIMAL(18,2) NULL,
    SecondGuarantorAmount DECIMAL(18,2) NULL,
    ThirdGuarantorAmount DECIMAL(18,2) NULL,
    
    -- Guarantor Group
    GuarantorGroupID INT NULL,
    GuarantorGroupName NVARCHAR(150) NULL,
    GuarantorSkippedAmount DECIMAL(18,2) NULL,
    
    -- Government & Administrative
    GovernmentRecordNumber NVARCHAR(100) NULL,
    Section NVARCHAR(150) NULL,
    Building NVARCHAR(150) NULL,
    [FILE] NVARCHAR(255) NULL,
    [T-Moh] NVARCHAR(100) NULL,
    
    -- Financial
    CostSample DECIMAL(18,2) NULL,
    
    -- Bed Type
    BedTypeID INT NULL,
    
    -- System Timestamp
    SystemTime DATETIME DEFAULT GETDATE()
)
GO

-- Create indexes for common search fields
CREATE INDEX IX_APP_RESERVE_PATIENT_FullName 
ON dbo.APP_RESERVE_PATIENT(FullName)
GO

CREATE INDEX IX_APP_RESERVE_PATIENT_FirstName 
ON dbo.APP_RESERVE_PATIENT(FirstName)
GO

CREATE INDEX IX_APP_RESERVE_PATIENT_LastName 
ON dbo.APP_RESERVE_PATIENT(LastName)
GO

CREATE INDEX IX_APP_RESERVE_PATIENT_DocumentNumber 
ON dbo.APP_RESERVE_PATIENT(DocumentNumber)
GO

CREATE INDEX IX_APP_RESERVE_PATIENT_MedicalFileNumber 
ON dbo.APP_RESERVE_PATIENT(MedicalFileNumber)
GO

CREATE INDEX IX_APP_RESERVE_PATIENT_SystemTime 
ON dbo.APP_RESERVE_PATIENT(SystemTime DESC)
GO

PRINT 'APP_RESERVE_PATIENT table created successfully!'
PRINT 'Total columns: 60 (matching APP_VIEWTABLE_PATIENT_ADMISSION)'
PRINT 'PatientAdmissionID starts at 100000 to avoid collisions'
GO

-- Verify table structure
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    CHARACTER_MAXIMUM_LENGTH,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'APP_RESERVE_PATIENT'
ORDER BY ORDINAL_POSITION
GO
