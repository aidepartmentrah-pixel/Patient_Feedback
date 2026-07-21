-- ============================================================
-- AIC-S7: Action Item Coordination — Notification Foundation
-- Purpose: Add acknowledgment tracking for the two action-item
--          notification cases:
--          1) Supervisor assigns an Action Item from the Calendar
--             (APP_SupervisorActionItem) -> add ack columns.
--          2) A higher-hierarchy person changes an Action Item
--             originally created at a lower level
--             (APP_SubcaseActionItem) -> new pending-notice table.
-- Safety: This script ONLY ADDS columns/tables. Does NOT modify
--         existing objects' behavior. Acknowledgment is a pure
--         side-channel flag, fully orthogonal to the existing
--         Status state machines on both tables.
-- Date: 2026-06-30
-- Safe to run multiple times (all checks are idempotent)
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'AIC-S7: Action Item Notification Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- 1) APP_SupervisorActionItem -- add acknowledgment columns
-- ============================================================
PRINT 'Adding acknowledgment columns to APP_SupervisorActionItem...';

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
    AND name = 'AcknowledgedAt'
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD AcknowledgedAt DATETIME NULL;

    PRINT '  [OK] AcknowledgedAt column added.';
END
ELSE
    PRINT '  [SKIP] AcknowledgedAt column already exists.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.columns
    WHERE object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
    AND name = 'AcknowledgedByUserID'
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD AcknowledgedByUserID INT NULL;

    PRINT '  [OK] AcknowledgedByUserID column added.';
END
ELSE
    PRINT '  [SKIP] AcknowledgedByUserID column already exists.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_SupervisorActionItem_AcknowledgedBy'
    AND parent_object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    ALTER TABLE dbo.APP_SupervisorActionItem
    ADD CONSTRAINT FK_SupervisorActionItem_AcknowledgedBy
        FOREIGN KEY (AcknowledgedByUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  [OK] FK to APP_Users (acknowledged by) added.';
END
ELSE
    PRINT '  [SKIP] FK to APP_Users (acknowledged by) already exists.';
GO

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_SupervisorActionItem_Unacknowledged'
    AND object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
)
BEGIN
    CREATE INDEX IX_SupervisorActionItem_Unacknowledged
    ON dbo.APP_SupervisorActionItem(TargetUserID, TargetOrgUnitID)
    WHERE AcknowledgedAt IS NULL;

    PRINT '  [OK] Filtered index on unacknowledged rows created.';
END
ELSE
    PRINT '  [SKIP] Filtered index on unacknowledged rows already exists.';
GO

-- ============================================================
-- 2) APP_SubcaseActionItemChangeNotice -- new table
-- One pending row per Action Item: upserted on every cross-user
-- edit, overwritten (not duplicated) until acknowledged.
-- ============================================================
PRINT '';
PRINT 'Creating TABLE: APP_SubcaseActionItemChangeNotice...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_SubcaseActionItemChangeNotice' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_SubcaseActionItemChangeNotice (
        NoticeID INT IDENTITY(1,1) PRIMARY KEY,

        ActionItemID INT NOT NULL,
        RecipientUserID INT NOT NULL,

        OldTitle NVARCHAR(300) NOT NULL,
        NewTitle NVARCHAR(300) NOT NULL,
        OldDescription NVARCHAR(MAX) NULL,
        NewDescription NVARCHAR(MAX) NULL,
        OldDueDate DATE NULL,
        NewDueDate DATE NULL,

        ChangedByUserID INT NOT NULL,
        ChangedAt DATETIME NOT NULL DEFAULT GETDATE(),

        AcknowledgedAt DATETIME NULL,
        AcknowledgedByUserID INT NULL
    );

    PRINT '  [OK] Table APP_SubcaseActionItemChangeNotice created.';
END
ELSE
    PRINT '  [SKIP] Table APP_SubcaseActionItemChangeNotice already exists.';
GO

-- ============================================================
-- FOREIGN KEY CONSTRAINTS -- APP_SubcaseActionItemChangeNotice
-- ============================================================
PRINT 'Adding Foreign Key Constraints for APP_SubcaseActionItemChangeNotice...';

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_ActionItemChangeNotice_ActionItem'
    AND parent_object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
)
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItemChangeNotice
    ADD CONSTRAINT FK_ActionItemChangeNotice_ActionItem
        FOREIGN KEY (ActionItemID)
        REFERENCES dbo.APP_SubcaseActionItem(ActionItemID);

    PRINT '  [OK] FK to APP_SubcaseActionItem added.';
END
ELSE
    PRINT '  [SKIP] FK to APP_SubcaseActionItem already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_ActionItemChangeNotice_Recipient'
    AND parent_object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
)
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItemChangeNotice
    ADD CONSTRAINT FK_ActionItemChangeNotice_Recipient
        FOREIGN KEY (RecipientUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  [OK] FK to APP_Users (recipient) added.';
END
ELSE
    PRINT '  [SKIP] FK to APP_Users (recipient) already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_ActionItemChangeNotice_ChangedBy'
    AND parent_object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
)
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItemChangeNotice
    ADD CONSTRAINT FK_ActionItemChangeNotice_ChangedBy
        FOREIGN KEY (ChangedByUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  [OK] FK to APP_Users (changed by) added.';
END
ELSE
    PRINT '  [SKIP] FK to APP_Users (changed by) already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.foreign_keys
    WHERE name = 'FK_ActionItemChangeNotice_AcknowledgedBy'
    AND parent_object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
)
BEGIN
    ALTER TABLE dbo.APP_SubcaseActionItemChangeNotice
    ADD CONSTRAINT FK_ActionItemChangeNotice_AcknowledgedBy
        FOREIGN KEY (AcknowledgedByUserID)
        REFERENCES dbo.APP_Users(UserID);

    PRINT '  [OK] FK to APP_Users (acknowledged by) added.';
END
ELSE
    PRINT '  [SKIP] FK to APP_Users (acknowledged by) already exists.';
GO

-- ============================================================
-- INDEXES -- APP_SubcaseActionItemChangeNotice
-- ============================================================
PRINT 'Creating Indexes...';

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

-- One unacknowledged notice per Action Item at a time (upsert target)
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'UX_ActionItemChangeNotice_PendingPerItem'
    AND object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
)
BEGIN
    CREATE UNIQUE INDEX UX_ActionItemChangeNotice_PendingPerItem
    ON dbo.APP_SubcaseActionItemChangeNotice(ActionItemID)
    WHERE AcknowledgedAt IS NULL;

    PRINT '  [OK] Unique filtered index (one pending notice per item) created.';
END
ELSE
    PRINT '  [SKIP] Unique filtered index already exists.';

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'IX_ActionItemChangeNotice_Recipient'
    AND object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
)
BEGIN
    CREATE INDEX IX_ActionItemChangeNotice_Recipient
    ON dbo.APP_SubcaseActionItemChangeNotice(RecipientUserID)
    WHERE AcknowledgedAt IS NULL;

    PRINT '  [OK] Filtered index on RecipientUserID created.';
END
ELSE
    PRINT '  [SKIP] Filtered index on RecipientUserID already exists.';
GO

PRINT '';
PRINT '============================================================';
PRINT 'AIC-S7: NOTIFICATION FOUNDATION COMPLETED';
PRINT '============================================================';
PRINT '';
PRINT 'Summary:';
PRINT '  - APP_SupervisorActionItem: +AcknowledgedAt, +AcknowledgedByUserID, +FK, +filtered index';
PRINT '  - APP_SubcaseActionItemChangeNotice: new table, 4 FKs, 2 filtered indexes';
PRINT '  - Important: NO CASCADE DELETE on any FK';
PRINT '  - Important: acknowledgment is orthogonal to existing Status state machines';
PRINT '';
PRINT 'Notification foundation is ready.';
PRINT '';

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '--- APP_SupervisorActionItem new columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_SupervisorActionItem')
AND name IN ('AcknowledgedAt', 'AcknowledgedByUserID');

PRINT '';
PRINT '--- APP_SubcaseActionItemChangeNotice columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice')
ORDER BY column_id;
