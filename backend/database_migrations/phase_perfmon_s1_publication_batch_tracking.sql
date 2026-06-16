-- ============================================================
-- PERFMON-S1: HCAT Performance & Delay Monitoring
--             Session 1 - Publication Batch Tracking
-- Date: 2026-06-15
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. No existing columns/tables altered.
-- Pure schema work - no workflow behavior changes.
-- ============================================================

USE IncidentManager;
GO

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

PRINT '============================================================';
PRINT 'PERFMON-S1: Publication Batch Tracking - Database Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- PART 1: APP_PublicationBatch
-- ============================================================
PRINT 'PART 1: Creating APP_PublicationBatch...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_PublicationBatch' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_PublicationBatch (
        PublicationBatchID INT IDENTITY(1,1) PRIMARY KEY,
        PublicationSerial INT NOT NULL,
        PublishedAt DATETIME NOT NULL DEFAULT GETDATE(),
        PublishedByUserID INT NOT NULL,
        CasesPublishedCount INT NOT NULL,
        Notes NVARCHAR(MAX) NULL,
        CreatedAt DATETIME NOT NULL DEFAULT GETDATE()
    );
    PRINT '  APP_PublicationBatch created';
END
ELSE
    PRINT '  APP_PublicationBatch already exists. Skipping.';

PRINT '';
PRINT 'PART 1b: Unique constraint + FK for APP_PublicationBatch...';

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_PublicationBatch_Serial' AND object_id = OBJECT_ID('dbo.APP_PublicationBatch'))
BEGIN
    CREATE UNIQUE INDEX UQ_PublicationBatch_Serial ON dbo.APP_PublicationBatch(PublicationSerial);
    PRINT '  UQ_PublicationBatch_Serial added';
END
ELSE
    PRINT '  UQ_PublicationBatch_Serial already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PublicationBatch_PublishedByUser')
BEGIN
    ALTER TABLE dbo.APP_PublicationBatch
    ADD CONSTRAINT FK_PublicationBatch_PublishedByUser
    FOREIGN KEY (PublishedByUserID) REFERENCES dbo.APP_Users(UserID);
    PRINT '  FK_PublicationBatch_PublishedByUser added';
END
ELSE
    PRINT '  FK_PublicationBatch_PublishedByUser already exists. Skipping.';

PRINT '';

-- ============================================================
-- PART 2: APP_PublicationBatchCase
-- ============================================================
PRINT 'PART 2: Creating APP_PublicationBatchCase...';

IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'APP_PublicationBatchCase' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    CREATE TABLE dbo.APP_PublicationBatchCase (
        PublicationBatchCaseID INT IDENTITY(1,1) PRIMARY KEY,
        PublicationBatchID INT NOT NULL,
        IncidentCaseID INT NOT NULL,
        AdministrativeSubcaseID INT NULL,
        TargetOrgUnitID INT NULL,
        CreatedAt DATETIME NOT NULL DEFAULT GETDATE()
    );
    PRINT '  APP_PublicationBatchCase created';
END
ELSE
    PRINT '  APP_PublicationBatchCase already exists. Skipping.';

PRINT '';
PRINT 'PART 2b: Foreign key + indexes for APP_PublicationBatchCase...';

-- Only the link back to the parent batch is constrained with a FK.
-- IncidentCaseID / AdministrativeSubcaseID / TargetOrgUnitID are deliberately
-- left without FK constraints: this table is a passive audit trail and must
-- never block deletes/cleanup on APP_IncidentCase, APP_AdministrativeSubcase
-- or AdminsrationUnit (consistent with the "passive recording only" rule).
IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PublicationBatchCase_Batch')
BEGIN
    ALTER TABLE dbo.APP_PublicationBatchCase
    ADD CONSTRAINT FK_PublicationBatchCase_Batch
    FOREIGN KEY (PublicationBatchID) REFERENCES dbo.APP_PublicationBatch(PublicationBatchID)
    ON DELETE CASCADE;
    PRINT '  FK_PublicationBatchCase_Batch added';
END
ELSE
    PRINT '  FK_PublicationBatchCase_Batch already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PublicationBatchCase_BatchID' AND object_id = OBJECT_ID('dbo.APP_PublicationBatchCase'))
BEGIN
    CREATE INDEX IX_PublicationBatchCase_BatchID ON dbo.APP_PublicationBatchCase(PublicationBatchID);
    PRINT '  IX_PublicationBatchCase_BatchID added';
END
ELSE
    PRINT '  IX_PublicationBatchCase_BatchID already exists. Skipping.';

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PublicationBatchCase_IncidentCaseID' AND object_id = OBJECT_ID('dbo.APP_PublicationBatchCase'))
BEGIN
    CREATE INDEX IX_PublicationBatchCase_IncidentCaseID ON dbo.APP_PublicationBatchCase(IncidentCaseID);
    PRINT '  IX_PublicationBatchCase_IncidentCaseID added';
END
ELSE
    PRINT '  IX_PublicationBatchCase_IncidentCaseID already exists. Skipping.';

PRINT '';

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '============================================================';
PRINT 'VERIFICATION';
PRINT '============================================================';
PRINT '';

PRINT '--- APP_PublicationBatch columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_PublicationBatch')
ORDER BY column_id;

PRINT '';
PRINT '--- APP_PublicationBatchCase columns ---';
SELECT name, TYPE_NAME(user_type_id) AS data_type, is_nullable
FROM sys.columns
WHERE object_id = OBJECT_ID('dbo.APP_PublicationBatchCase')
ORDER BY column_id;

PRINT '';
PRINT '--- Constraints/Indexes ---';
SELECT name AS object_name, 'FK' AS object_type FROM sys.foreign_keys
WHERE parent_object_id IN (OBJECT_ID('dbo.APP_PublicationBatch'), OBJECT_ID('dbo.APP_PublicationBatchCase'))
UNION ALL
SELECT name, 'INDEX' FROM sys.indexes
WHERE object_id IN (OBJECT_ID('dbo.APP_PublicationBatch'), OBJECT_ID('dbo.APP_PublicationBatchCase'))
  AND name IS NOT NULL;

PRINT '';
PRINT '============================================================';
PRINT 'PERFMON-S1: Migration complete.';
PRINT 'Expected: 2 new tables (APP_PublicationBatch, APP_PublicationBatchCase),';
PRINT '          1 unique index + 1 FK on APP_PublicationBatch,';
PRINT '          1 FK + 2 indexes on APP_PublicationBatchCase.';
PRINT '============================================================';
