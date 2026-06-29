-- ============================================================
-- AIC-S1: Action Item Coordination (Iteration 4) — Database Foundation
-- Purpose: Create APP_SupervisorActionItem + APP_SupervisorActionItemAuditLog
-- Safety: This script ONLY ADDS new tables. Does NOT modify existing objects.
-- Scope: Administrative, cross-org-unit Action Items created by Complaint Supervisor.
--        Deliberately separate from APP_SubcaseActionItem — that table's status
--        machine is load-bearing for case-escalation gating (see follow_up_service.py)
--        and must not be touched by this feature.
-- Date: 2026-06-24
-- Safe to run multiple times (all checks are idempotent)
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'AIC-S1: Action Item Coordination — Database Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- TABLE: APP_SupervisorActionItem
-- ============================================================
PRINT 'Creating TABLE: APP_SupervisorActionItem...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_SupervisorActionItem' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_SupervisorActionItem (
        -- Identity
        ActionItemID INT IDENTITY(1,1) PRIMARY KEY,

        -- Case Link (always required — every administrative action item traces back to a complaint)
        IncidentRequestCaseID INT NOT NULL,

        -- Subcase Link (optional — the target unit may not have an existing subcase for this case)
        SubcaseID INT NULL,

        -- Target (where the work is being routed TO — may be outside the case's own chain)
        TargetOrgUnitID INT NOT NULL,
        TargetUserID INT NULL,

        -- Creator (who imposed this assignment, and in what role at the time)
        CreatedByUserID INT NOT NULL,
        CreatedByRoleCode NVARCHAR(50) NOT NULL,

        -- Business Data
        Description NVARCHAR(MAX) NOT NULL,
        DueDate DATE NULL,

        -- Status: administrative only — no acceptance/proposal workflow in this iteration
        Status NVARCHAR(20) NOT NULL DEFAULT 'PENDING',

        -- Lifecycle Timestamps
        CreatedAt DATETIME NOT NULL DEFAULT GETDATE(),
        CompletedAt DATETIME NULL,
        CancelledAt DATETIME NULL,
        UpdatedAt DATETIME NULL,
        UpdatedByUserID INT NULL,

        CONSTRAINT CK_SupervisorActionItem_Status
            CHECK (Status IN ('PENDING', 'COMPLETED', 'CANCELLED'))
    );

    PRINT '✅ Table APP_SupervisorActionItem created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_SupervisorActionItem already exists. Skipping creation.';
END
GO

-- ============================================================
-- TABLE: APP_SupervisorActionItemAuditLog
-- ============================================================
PRINT 'Creating TABLE: APP_SupervisorActionItemAuditLog...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_SupervisorActionItemAuditLog' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_SupervisorActionItemAuditLog (
        AuditLogID INT IDENTITY(1,1) PRIMARY KEY,
        ActionItemID INT NOT NULL,
        Action NVARCHAR(20) NOT NULL,
        PerformedByUserID INT NOT NULL,
        PerformedAt DATETIME NOT NULL DEFAULT GETDATE(),
        Note NVARCHAR(500) NULL,

        CONSTRAINT CK_SupervisorActionItemAuditLog_Action
            CHECK (Action IN ('CREATED', 'COMPLETED', 'CANCELLED'))
    );

    PRINT '✅ Table APP_SupervisorActionItemAuditLog created successfully.';
END
ELSE
BEGIN
    PRINT '⚠️  Table APP_SupervisorActionItemAuditLog already exists. Skipping creation.';
END
GO

-- ============================================================
-- FOREIGN KEY CONSTRAINTS — APP_SupervisorActionItem
-- ============================================================
PRINT 'Adding Foreign Key Constraints for APP_SupervisorActionItem...';

-- FK to APP_IncidentCase (NO CASCADE DELETE - preserve audit history)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_Case'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_Case
        FOREIGN KEY (IncidentRequestCaseID)
        REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID);

    PRINT '  ✅ FK to APP_IncidentCase added (NO CASCADE DELETE).';
END
ELSE
    PRINT '  ⚠️  FK to APP_IncidentCase already exists.';

-- FK to APP_AdministrativeSubcase (optional link, NO CASCADE DELETE)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_Subcase'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_Subcase
        FOREIGN KEY (SubcaseID)
        REFERENCES dbo.APP_AdministrativeSubcase(SubcaseID);

    PRINT '  ✅ FK to APP_AdministrativeSubcase added (NO CASCADE DELETE).';
END
ELSE
    PRINT '  ⚠️  FK to APP_AdministrativeSubcase already exists.';

-- FK to AdminsrationUnit (target org unit)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_TargetOrgUnit'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_TargetOrgUnit
        FOREIGN KEY (TargetOrgUnitID)
        REFERENCES dbo.AdminsrationUnit(UniqueID);

    PRINT '  ✅ FK to AdminsrationUnit (target) added.';
END
ELSE
    PRINT '  ⚠️  FK to AdminsrationUnit (target) already exists.';

-- FK to APP_Users (target user, optional)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_TargetUser'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_TargetUser
        FOREIGN KEY (TargetUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  ✅ FK to APP_Users (target user) added.';
END
ELSE
    PRINT '  ⚠️  FK to APP_Users (target user) already exists.';

-- FK to APP_Users (creator)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_CreatedBy'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_CreatedBy
        FOREIGN KEY (CreatedByUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  ✅ FK to APP_Users (creator) added.';
END
ELSE
    PRINT '  ⚠️  FK to APP_Users (creator) already exists.';

-- FK to APP_Users (updater, optional)
IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_UpdatedBy'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_UpdatedBy
        FOREIGN KEY (UpdatedByUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  ✅ FK to APP_Users (updater) added.';
END
ELSE
    PRINT '  ⚠️  FK to APP_Users (updater) already exists.';
GO

-- ============================================================
-- FOREIGN KEY CONSTRAINTS — APP_SupervisorActionItemAuditLog
-- ============================================================
PRINT 'Adding Foreign Key Constraints for APP_SupervisorActionItemAuditLog...';

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItemAuditLog_ActionItem'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItemAuditLog')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItemAuditLog
    ADD CONSTRAINT FK_SupervisorActionItemAuditLog_ActionItem
        FOREIGN KEY (ActionItemID)
        REFERENCES dbo.APP_SupervisorActionItem(ActionItemID);

    PRINT '  ✅ FK to APP_SupervisorActionItem added (NO CASCADE DELETE).';
END
ELSE
    PRINT '  ⚠️  FK to APP_SupervisorActionItem already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItemAuditLog_PerformedBy'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItemAuditLog')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItemAuditLog
    ADD CONSTRAINT FK_SupervisorActionItemAuditLog_PerformedBy
        FOREIGN KEY (PerformedByUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  ✅ FK to APP_Users (performed by) added.';
END
ELSE
    PRINT '  ⚠️  FK to APP_Users (performed by) already exists.';
GO

-- ============================================================
-- INDEXES
-- ============================================================
PRINT 'Creating Indexes...';

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_SupervisorActionItem_TargetOrgUnit'
    AND object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    CREATE INDEX IX_SupervisorActionItem_TargetOrgUnit
    ON dbo.APP_SupervisorActionItem(TargetOrgUnitID);

    PRINT '  ✅ Index on TargetOrgUnitID created.';
END
ELSE
    PRINT '  ⚠️  Index on TargetOrgUnitID already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_SupervisorActionItem_Status'
    AND object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    CREATE INDEX IX_SupervisorActionItem_Status
    ON dbo.APP_SupervisorActionItem(Status);

    PRINT '  ✅ Index on Status created.';
END
ELSE
    PRINT '  ⚠️  Index on Status already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_SupervisorActionItem_Case'
    AND object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    CREATE INDEX IX_SupervisorActionItem_Case
    ON dbo.APP_SupervisorActionItem(IncidentRequestCaseID);

    PRINT '  ✅ Index on IncidentRequestCaseID created.';
END
ELSE
    PRINT '  ⚠️  Index on IncidentRequestCaseID already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_SupervisorActionItem_TargetUser'
    AND object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    CREATE INDEX IX_SupervisorActionItem_TargetUser
    ON dbo.APP_SupervisorActionItem(TargetUserID)
    WHERE TargetUserID IS NOT NULL;

    PRINT '  ✅ Index on TargetUserID created (filtered).';
END
ELSE
    PRINT '  ⚠️  Index on TargetUserID already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_SupervisorActionItemAuditLog_ActionItem'
    AND object_id = OBJECT_ID('dbo.APP_SupervisorActionItemAuditLog')
)
BEGIN
    CREATE INDEX IX_SupervisorActionItemAuditLog_ActionItem
    ON dbo.APP_SupervisorActionItemAuditLog(ActionItemID);

    PRINT '  ✅ Index on AuditLog.ActionItemID created.';
END
ELSE
    PRINT '  ⚠️  Index on AuditLog.ActionItemID already exists.';
GO

PRINT '';
PRINT '============================================================';
PRINT 'AIC-S1: SUPERVISOR ACTION ITEM FOUNDATION COMPLETED';
PRINT '============================================================';
PRINT '';
PRINT 'Summary:';
PRINT '  - Table: APP_SupervisorActionItem created';
PRINT '  - Table: APP_SupervisorActionItemAuditLog created';
PRINT '  - Foreign Keys: 6 on APP_SupervisorActionItem, 2 on AuditLog';
PRINT '  - Indexes: 4 on APP_SupervisorActionItem, 1 on AuditLog';
PRINT '  - Important: NO CASCADE DELETE on any FK';
PRINT '  - Important: completely separate from APP_SubcaseActionItem';
PRINT '';
PRINT '✅ Supervisor Action Item foundation is ready.';
PRINT '';

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '--- APP_SupervisorActionItem columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
ORDER BY column_id;

PRINT '';
PRINT '--- APP_SupervisorActionItemAuditLog columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_SupervisorActionItemAuditLog')
ORDER BY column_id;
