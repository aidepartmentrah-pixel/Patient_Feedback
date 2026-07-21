-- ============================================================
-- PHASE ML-S8: ML Architecture Consolidation — Historical Migration Schema
-- Purpose: Small additive schema needed before the Stage 8 historical
--          migration script (backend/scripts/ml_stage8_historical_migration.py)
--          can run:
--            1. A filtered UNIQUE index on ml.HistoricalTrainingExample so a
--               re-run of the migration script cannot insert the same
--               legacy SQLite row twice (same defense-in-depth pattern as
--               dbo.APP_DataMigration_Map / ml.ImportSourceRecordMap).
--            2. Three small archival tables for the legacy SQLite
--               training-run history (training_runs / model_metrics /
--               ml_db_size_history) so that history isn't lost when
--               SQLite is eventually retired (Stage 13). Straight 1:1
--               copies — Stage 9 introduces the new forward-looking
--               run-tracking design, this is not that.
-- Safety: Additive only. Does not modify any existing dbo.* or ml.* table.
-- Reference: ML_ARCHITECTURE_DECISION_RECORD.md, Workstream 2 Stage 8
-- Date: 2026-07-16
-- Safe to run multiple times (all checks are idempotent)
-- ============================================================

USE IncidentManager;
GO

SET QUOTED_IDENTIFIER ON;
GO

PRINT '============================================================';
PRINT 'PHASE ML-S8: Historical Migration Schema';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- INDEX: filtered UNIQUE on ml.HistoricalTrainingExample
-- Purpose: Row-level idempotency for the Stage 8 migration script — the
-- same (LegacySourceTable, LegacySourceRowID) pair can never be inserted
-- twice. Filtered (not a plain unique constraint) because both columns
-- are nullable and only rows actually tagged with a legacy source need
-- this guarantee.
-- ============================================================
IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'UQ_ml_HistoricalTrainingExample_LegacySource'
      AND object_id = OBJECT_ID('ml.HistoricalTrainingExample')
)
BEGIN
    PRINT 'Creating INDEX: UQ_ml_HistoricalTrainingExample_LegacySource...';

    CREATE UNIQUE NONCLUSTERED INDEX UQ_ml_HistoricalTrainingExample_LegacySource
        ON ml.HistoricalTrainingExample (LegacySourceTable, LegacySourceRowID)
        WHERE LegacySourceTable IS NOT NULL AND LegacySourceRowID IS NOT NULL;

    PRINT 'UQ_ml_HistoricalTrainingExample_LegacySource created successfully';
END
ELSE
BEGIN
    PRINT 'UQ_ml_HistoricalTrainingExample_LegacySource already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.LegacyTrainingRunHistory
-- Purpose: 1:1 archival copy of the legacy SQLite training_runs table.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'LegacyTrainingRunHistory' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.LegacyTrainingRunHistory...';

    CREATE TABLE ml.LegacyTrainingRunHistory (
        LegacyTrainingRunHistoryID INT IDENTITY(1,1) NOT NULL,
        RunID NVARCHAR(100) NOT NULL,
        StartedAt NVARCHAR(50) NULL,
        FinishedAt NVARCHAR(50) NULL,
        Status NVARCHAR(50) NULL,
        ModelsTrained INT NULL,
        LegacyCreatedAt NVARCHAR(50) NULL,
        MigratedAt DATETIME2 NOT NULL DEFAULT GETDATE(),

        CONSTRAINT PK_ml_LegacyTrainingRunHistory PRIMARY KEY CLUSTERED (LegacyTrainingRunHistoryID),
        CONSTRAINT UQ_ml_LegacyTrainingRunHistory_RunID UNIQUE (RunID)
    );

    PRINT 'ml.LegacyTrainingRunHistory created successfully';
END
ELSE
BEGIN
    PRINT 'ml.LegacyTrainingRunHistory already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.LegacyModelMetricHistory
-- Purpose: 1:1 archival copy of the legacy SQLite model_metrics table.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'LegacyModelMetricHistory' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.LegacyModelMetricHistory...';

    CREATE TABLE ml.LegacyModelMetricHistory (
        LegacyModelMetricHistoryID INT IDENTITY(1,1) NOT NULL,
        LegacyMetricID INT NULL,
        RunID NVARCHAR(100) NULL,
        ModelName NVARCHAR(200) NULL,
        NumRecords INT NULL,
        Accuracy FLOAT NULL,
        Precision_ FLOAT NULL,
        Recall_ FLOAT NULL,
        F1 FLOAT NULL,
        LastTrained NVARCHAR(50) NULL,
        MigratedAt DATETIME2 NOT NULL DEFAULT GETDATE(),

        CONSTRAINT PK_ml_LegacyModelMetricHistory PRIMARY KEY CLUSTERED (LegacyModelMetricHistoryID)
    );

    PRINT 'ml.LegacyModelMetricHistory created successfully';
END
ELSE
BEGIN
    PRINT 'ml.LegacyModelMetricHistory already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.LegacyDbSizeHistory
-- Purpose: 1:1 archival copy of the legacy SQLite ml_db_size_history table.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'LegacyDbSizeHistory' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.LegacyDbSizeHistory...';

    CREATE TABLE ml.LegacyDbSizeHistory (
        LegacyDbSizeHistoryID INT IDENTITY(1,1) NOT NULL,
        RecordDate NVARCHAR(50) NOT NULL,
        RecordCount INT NOT NULL,
        LegacyRecordedAt NVARCHAR(50) NULL,
        MigratedAt DATETIME2 NOT NULL DEFAULT GETDATE(),

        CONSTRAINT PK_ml_LegacyDbSizeHistory PRIMARY KEY CLUSTERED (LegacyDbSizeHistoryID),
        CONSTRAINT UQ_ml_LegacyDbSizeHistory_RecordDate UNIQUE (RecordDate)
    );

    PRINT 'ml.LegacyDbSizeHistory created successfully';
END
ELSE
BEGIN
    PRINT 'ml.LegacyDbSizeHistory already exists — skipping';
END
GO

-- Record this migration itself
IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = 'phase_ml_s8_historical_migration_schema')
BEGIN
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, AppliedBy, Success)
    VALUES ('phase_ml_s8_historical_migration_schema', SYSTEM_USER, 1);
    PRINT 'Recorded this migration in dbo.SchemaMigrationHistory';
END
GO

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '';
PRINT '============================================================';
PRINT 'PHASE ML-S8 SCHEMA VERIFICATION';
PRINT '============================================================';

SELECT
    s.name AS SchemaName,
    t.name AS TableName,
    p.rows AS ApproxRowCount
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
WHERE t.name IN ('LegacyTrainingRunHistory', 'LegacyModelMetricHistory', 'LegacyDbSizeHistory')
ORDER BY s.name, t.name;

SELECT name AS IndexName, is_unique AS IsUnique
FROM sys.indexes
WHERE object_id = OBJECT_ID('ml.HistoricalTrainingExample') AND name = 'UQ_ml_HistoricalTrainingExample_LegacySource';

PRINT '';
PRINT 'Phase ML-S8 schema complete. No existing dbo.*/ml.* table was modified.';
PRINT '============================================================';
GO
