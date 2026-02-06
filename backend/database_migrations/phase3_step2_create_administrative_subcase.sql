-- ============================================================
-- PHASE 3 - STEP 3.2: CREATE ADMINISTRATIVE_SUBCASE TABLE
-- ============================================================
-- Purpose: Create the core workflow table for Phase 3 parallel system
-- Safety: This script ONLY ADDS a new table. Does NOT modify existing objects.
-- Date: 2026-01-29
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'PHASE 3 - STEP 3.2: CREATING ADMINISTRATIVE_SUBCASE TABLE';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- TABLE: APP_AdministrativeSubcase
-- ============================================================
PRINT 'Creating TABLE: APP_AdministrativeSubcase...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_AdministrativeSubcase' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_AdministrativeSubcase (
        -- Identity
        SubcaseID INT IDENTITY(1,1) PRIMARY KEY,
        
        -- Type & Parent Link
        CaseType NVARCHAR(50) NOT NULL,
        IncidentRequestCaseID INT NULL,
        SeasonalReportID INT NULL,
        
        -- Target Org Unit (the department/section this subcase is about)
        TargetOrgUnitID INT NOT NULL,
        
        -- Workflow Status
        Status NVARCHAR(50) NOT NULL,
        
        -- Text Payloads (explicit at each workflow level)
        SectionExplanationText NVARCHAR(MAX) NULL,
        SectionRejectionText NVARCHAR(MAX) NULL,
        DepartmentExplanationText NVARCHAR(MAX) NULL,
        DepartmentRejectionText NVARCHAR(MAX) NULL,
        AdministrationExplanationText NVARCHAR(MAX) NULL,
        AdministrationRejectionText NVARCHAR(MAX) NULL,
        
        -- Audit Fields
        CreatedAt DATETIME NOT NULL DEFAULT GETDATE(),
        CreatedByUserID INT NOT NULL,
        UpdatedAt DATETIME NULL,
        UpdatedByUserID INT NULL,
        
        -- Business Rule: Either IncidentRequestCaseID OR SeasonalReportID must be non-null
        CONSTRAINT CK_AdministrativeSubcase_ParentLink 
            CHECK (
                (IncidentRequestCaseID IS NOT NULL AND SeasonalReportID IS NULL) OR
                (IncidentRequestCaseID IS NULL AND SeasonalReportID IS NOT NULL)
            )
    );
    
    PRINT '✅ Table APP_AdministrativeSubcase created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_AdministrativeSubcase already exists. Skipping creation.';
END
GO

-- ============================================================
-- FOREIGN KEY CONSTRAINTS
-- ============================================================
PRINT 'Adding Foreign Key Constraints...';

-- FK to APP_IncidentCase
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys 
    WHERE name = 'FK_AdministrativeSubcase_IncidentCase' 
    AND parent_object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_IncidentCase
        FOREIGN KEY (IncidentRequestCaseID)
        REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID)
        ON DELETE CASCADE;
    
    PRINT '  ✅ FK to APP_IncidentCase added.';
END
ELSE
BEGIN
    PRINT '  ⚠️  FK to APP_IncidentCase already exists.';
END

-- FK to APP_SeasonalOrgUnitReport
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys 
    WHERE name = 'FK_AdministrativeSubcase_SeasonalReport' 
    AND parent_object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_SeasonalReport
        FOREIGN KEY (SeasonalReportID)
        REFERENCES dbo.APP_SeasonalOrgUnitReport(SeasonalReportID)
        ON DELETE CASCADE;
    
    PRINT '  ✅ FK to APP_SeasonalOrgUnitReport added.';
END
ELSE
BEGIN
    PRINT '  ⚠️  FK to APP_SeasonalOrgUnitReport already exists.';
END

-- FK to AdminsrationUnit
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys 
    WHERE name = 'FK_AdministrativeSubcase_OrgUnit' 
    AND parent_object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
    ADD CONSTRAINT FK_AdministrativeSubcase_OrgUnit
        FOREIGN KEY (TargetOrgUnitID)
        REFERENCES dbo.AdminsrationUnit(UniqueID);
    
    PRINT '  ✅ FK to AdminsrationUnit added.';
END
ELSE
BEGIN
    PRINT '  ⚠️  FK to AdminsrationUnit already exists.';
END
GO

-- ============================================================
-- INDEXES
-- ============================================================
PRINT 'Creating Indexes...';

-- Set options for filtered indexes
SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

-- Index on Status for workflow queries
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_AdministrativeSubcase_Status' 
    AND object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    CREATE INDEX IX_AdministrativeSubcase_Status
    ON dbo.APP_AdministrativeSubcase(Status);
    
    PRINT '  ✅ Index on Status created.';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on Status already exists.';
END

-- Index on CaseType + IncidentRequestCaseID for incident-based queries
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_AdministrativeSubcase_Incident' 
    AND object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    CREATE INDEX IX_AdministrativeSubcase_Incident
    ON dbo.APP_AdministrativeSubcase(CaseType, IncidentRequestCaseID)
    WHERE IncidentRequestCaseID IS NOT NULL;
    
    PRINT '  ✅ Index on CaseType + IncidentRequestCaseID created.';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on CaseType + IncidentRequestCaseID already exists.';
END

-- Index on CaseType + SeasonalReportID for seasonal-based queries
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_AdministrativeSubcase_Seasonal' 
    AND object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    CREATE INDEX IX_AdministrativeSubcase_Seasonal
    ON dbo.APP_AdministrativeSubcase(CaseType, SeasonalReportID)
    WHERE SeasonalReportID IS NOT NULL;
    
    PRINT '  ✅ Index on CaseType + SeasonalReportID created.';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on CaseType + SeasonalReportID already exists.';
END

-- Index on TargetOrgUnitID for org-unit-based queries (inbox, dashboards)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_AdministrativeSubcase_TargetOrgUnit' 
    AND object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    CREATE INDEX IX_AdministrativeSubcase_TargetOrgUnit
    ON dbo.APP_AdministrativeSubcase(TargetOrgUnitID);
    
    PRINT '  ✅ Index on TargetOrgUnitID created.';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on TargetOrgUnitID already exists.';
END
GO

PRINT '';
PRINT '============================================================';
PRINT 'PHASE 3 - STEP 3.2: ADMINISTRATIVE_SUBCASE TABLE COMPLETED';
PRINT '============================================================';
PRINT '';
PRINT 'Summary:';
PRINT '  - Table: APP_AdministrativeSubcase created';
PRINT '  - Foreign Keys: 3 (Incident, Seasonal, OrgUnit)';
PRINT '  - Indexes: 4 (Status, Incident, Seasonal, OrgUnit)';
PRINT '  - Check Constraint: Parent link validation';
PRINT '';
PRINT '✅ Administrative Subcase table is ready for Phase 3 workflow.';
PRINT '';
