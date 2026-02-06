-- ============================================================
-- PHASE 3 - STEP 3.3: CREATE ACTION_ITEM_SUBCASE TABLE
-- ============================================================
-- Purpose: Create action item tracking table for Phase 3 workflow system
-- Safety: This script ONLY ADDS a new table. Does NOT modify existing objects.
-- Date: 2026-01-29
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'PHASE 3 - STEP 3.3: CREATING ACTION_ITEM_SUBCASE TABLE';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- TABLE: APP_SubcaseActionItem
-- ============================================================
PRINT 'Creating TABLE: APP_SubcaseActionItem...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_SubcaseActionItem' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_SubcaseActionItem (
        -- Identity
        ActionItemID INT IDENTITY(1,1) PRIMARY KEY,
        
        -- Parent Link
        SubcaseID INT NOT NULL,
        
        -- Workflow Status
        Status NVARCHAR(50) NOT NULL,
        
        -- Business Data
        Title NVARCHAR(300) NOT NULL,
        Description NVARCHAR(MAX) NULL,
        DueDate DATE NULL,
        
        -- Assignment (future-proof, not enforced yet)
        AssignedToUserID INT NULL,
        
        -- Execution Tracking
        StartedAt DATETIME NULL,
        CompletedAt DATETIME NULL,
        VerifiedAt DATETIME NULL,
        
        -- Audit Fields
        CreatedAt DATETIME NOT NULL DEFAULT GETDATE(),
        CreatedByUserID INT NOT NULL,
        UpdatedAt DATETIME NULL,
        UpdatedByUserID INT NULL
    );
    
    PRINT '✅ Table APP_SubcaseActionItem created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_SubcaseActionItem already exists. Skipping creation.';
END
GO

-- ============================================================
-- FOREIGN KEY CONSTRAINTS
-- ============================================================
PRINT 'Adding Foreign Key Constraints...';

-- FK to APP_AdministrativeSubcase (NO CASCADE DELETE - prevent accidental data loss)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys 
    WHERE name = 'FK_SubcaseActionItem_Subcase' 
    AND parent_object_id = OBJECT_ID('dbo.APP_SubcaseActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItem
    ADD CONSTRAINT FK_SubcaseActionItem_Subcase
        FOREIGN KEY (SubcaseID)
        REFERENCES dbo.APP_AdministrativeSubcase(SubcaseID);
    
    PRINT '  ✅ FK to APP_AdministrativeSubcase added (NO CASCADE DELETE).';
END
ELSE
BEGIN
    PRINT '  ⚠️  FK to APP_AdministrativeSubcase already exists.';
END

-- FK to APP_Lookup_SubcaseActionItemStatus
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys 
    WHERE name = 'FK_SubcaseActionItem_Status' 
    AND parent_object_id = OBJECT_ID('dbo.APP_SubcaseActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItem
    ADD CONSTRAINT FK_SubcaseActionItem_Status
        FOREIGN KEY (Status)
        REFERENCES dbo.APP_Lookup_SubcaseActionItemStatus(StatusCode);
    
    PRINT '  ✅ FK to APP_Lookup_SubcaseActionItemStatus added.';
END
ELSE
BEGIN
    PRINT '  ⚠️  FK to APP_Lookup_SubcaseActionItemStatus already exists.';
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

-- Index on SubcaseID for parent-based queries
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_SubcaseActionItem_Subcase' 
    AND object_id = OBJECT_ID('dbo.APP_SubcaseActionItem')
)
BEGIN
    CREATE INDEX IX_SubcaseActionItem_Subcase
    ON dbo.APP_SubcaseActionItem(SubcaseID);
    
    PRINT '  ✅ Index on SubcaseID created.';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on SubcaseID already exists.';
END

-- Index on Status for workflow queries
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_SubcaseActionItem_Status' 
    AND object_id = OBJECT_ID('dbo.APP_SubcaseActionItem')
)
BEGIN
    CREATE INDEX IX_SubcaseActionItem_Status
    ON dbo.APP_SubcaseActionItem(Status);
    
    PRINT '  ✅ Index on Status created.';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on Status already exists.';
END

-- Index on AssignedToUserID for user-based queries (future)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes 
    WHERE name = 'IX_SubcaseActionItem_AssignedUser' 
    AND object_id = OBJECT_ID('dbo.APP_SubcaseActionItem')
)
BEGIN
    CREATE INDEX IX_SubcaseActionItem_AssignedUser
    ON dbo.APP_SubcaseActionItem(AssignedToUserID)
    WHERE AssignedToUserID IS NOT NULL;
    
    PRINT '  ✅ Index on AssignedToUserID created (filtered).';
END
ELSE
BEGIN
    PRINT '  ⚠️  Index on AssignedToUserID already exists.';
END
GO

PRINT '';
PRINT '============================================================';
PRINT 'PHASE 3 - STEP 3.3: ACTION_ITEM_SUBCASE TABLE COMPLETED';
PRINT '============================================================';
PRINT '';
PRINT 'Summary:';
PRINT '  - Table: APP_SubcaseActionItem created';
PRINT '  - Foreign Keys: 2 (Subcase, Status)';
PRINT '  - Indexes: 3 (SubcaseID, Status, AssignedToUserID)';
PRINT '  - Important: NO CASCADE DELETE on Subcase FK';
PRINT '';
PRINT '✅ Subcase Action Item table is ready for Phase 3 workflow.';
PRINT '';
