-- ============================================================
-- PHASE ML-S1: ML Architecture Consolidation — Schema Foundation
-- Purpose: Create the `ml` schema and its tables inside IncidentManager,
--          consolidating ML training data currently held in a standalone
--          SQLite file (models_directory/patient_feedback_ml.db) into the
--          same database as the operational data, with real FK/UNIQUE
--          constraints the old file-based store could never enforce.
-- Safety: This script ONLY ADDS new schema/tables. It does NOT modify,
--         read from, or write to any existing dbo.* table, and no
--         application code references these new tables yet (see
--         ML_ARCHITECTURE_DECISION_RECORD.md, Workstream 1 Stage 2).
-- Reference: ML_ARCHITECTURE_DECISION_RECORD.md section 6 (logical data model)
-- Date: 2026-07-16
-- Safe to run multiple times (all checks are idempotent)
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'PHASE ML-S1: ML Schema Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- SCHEMA: ml
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'ml')
BEGIN
    EXEC('CREATE SCHEMA ml');
    PRINT 'Created SCHEMA: ml';
END
ELSE
BEGIN
    PRINT 'SCHEMA ml already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.EmbeddingModelVersion
-- Purpose: Never trust a model label alone — the existing code calls the
-- live embedding model "MPNet" while its own saved config identifies it as
-- XLMRobertaModel. This table lets every stored embedding be traced to the
-- exact model/version/config that produced it.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'EmbeddingModelVersion' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.EmbeddingModelVersion...';

    CREATE TABLE ml.EmbeddingModelVersion (
        EmbeddingModelVersionID INT IDENTITY(1,1) NOT NULL,
        ModelName NVARCHAR(200) NOT NULL,
        ModelPathOrIdentifier NVARCHAR(500) NOT NULL,
        ModelArchitecture NVARCHAR(200) NULL,
        ModelChecksum NVARCHAR(128) NULL,
        EmbeddingDimension INT NOT NULL,
        PoolingMethod NVARCHAR(50) NULL,
        NormalizationMethod NVARCHAR(50) NULL,
        TokenizerIdentifier NVARCHAR(200) NULL,
        ActivatedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        RetiredAt DATETIME2 NULL,
        IsActive BIT NOT NULL DEFAULT 1,
        ConfigurationJson NVARCHAR(MAX) NULL,

        CONSTRAINT PK_ml_EmbeddingModelVersion PRIMARY KEY CLUSTERED (EmbeddingModelVersionID)
    );

    PRINT 'ml.EmbeddingModelVersion created successfully';
END
ELSE
BEGIN
    PRINT 'ml.EmbeddingModelVersion already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.ImportBatch
-- Purpose: One row per uploaded import file — batch-level tracking and
-- file-checksum duplicate detection.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ImportBatch' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.ImportBatch...';

    CREATE TABLE ml.ImportBatch (
        ImportBatchID INT IDENTITY(1,1) NOT NULL,
        OriginalFileName NVARCHAR(500) NULL,
        FileChecksum NVARCHAR(128) NULL,
        TemplateVersion NVARCHAR(50) NULL,
        UploadedByUserID INT NULL,
        UploadedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        Status NVARCHAR(20) NOT NULL DEFAULT 'Processing',
        TotalRows INT NULL,
        AcceptedRows INT NULL,
        RejectedRows INT NULL,
        DuplicateRows INT NULL,
        CreatedCaseCount INT NULL,
        MLCompletedCount INT NULL,
        MLFailedCount INT NULL,
        CompletedAt DATETIME2 NULL,

        CONSTRAINT PK_ml_ImportBatch PRIMARY KEY CLUSTERED (ImportBatchID),
        CONSTRAINT CK_ml_ImportBatch_Status CHECK (Status IN ('Processing', 'Completed', 'Failed')),
        CONSTRAINT FK_ml_ImportBatch_User
            FOREIGN KEY (UploadedByUserID)
            REFERENCES dbo.APP_Users(UserID)
    );

    CREATE NONCLUSTERED INDEX IX_ml_ImportBatch_FileChecksum ON ml.ImportBatch (FileChecksum);

    PRINT 'ml.ImportBatch created successfully';
END
ELSE
BEGIN
    PRINT 'ml.ImportBatch already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.CaseTrainingRecord
-- Purpose: The current ML training representation of an operational case.
-- One row per case (UNIQUE constraint) — edits update this row in place;
-- they never append a duplicate (replacing the old SQLite adapter's
-- append-only behavior).
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'CaseTrainingRecord' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.CaseTrainingRecord...';

    CREATE TABLE ml.CaseTrainingRecord (
        CaseTrainingRecordID INT IDENTITY(1,1) NOT NULL,
        IncidentRequestCaseID INT NOT NULL,

        ComplaintText NVARCHAR(MAX) NULL,
        ImmediateActionText NVARCHAR(MAX) NULL,
        TakenActionText NVARCHAR(MAX) NULL,

        FeedbackTypeID INT NULL,
        DomainID INT NULL,
        CategoryID INT NULL,
        SubCategoryID INT NULL,
        ClassificationID INT NULL,
        SeverityLevelID INT NULL,
        StageID INT NULL,
        HarmLevelID INT NULL,
        ImprovementOpportunityTypeID INT NULL,

        -- Only the two confirmed load-bearing embeddings are actively
        -- maintained going forward (see ML_ARCHITECTURE_DECISION_RECORD.md
        -- section 4.5) — complaint text alone, and complaint+immediate+taken
        -- combined. Stored as raw float32 bytes (VARBINARY), matching the
        -- representation already produced by get_embedding()/get_embedding_list().
        ComplaintEmbedding VARBINARY(MAX) NULL,
        CombinedTextEmbedding VARBINARY(MAX) NULL,

        EmbeddingModelVersionID INT NULL,
        EmbeddingDimension INT NULL,

        ProcessingStatus NVARCHAR(20) NOT NULL DEFAULT 'Pending',
        LastProcessedAt DATETIME2 NULL,
        SourceDataUpdatedAt DATETIME2 NULL,
        CreatedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        UpdatedAt DATETIME2 NULL,

        CONSTRAINT PK_ml_CaseTrainingRecord PRIMARY KEY CLUSTERED (CaseTrainingRecordID),
        CONSTRAINT UQ_ml_CaseTrainingRecord_Case UNIQUE (IncidentRequestCaseID),
        CONSTRAINT CK_ml_CaseTrainingRecord_Status
            CHECK (ProcessingStatus IN ('Pending', 'Processing', 'Completed', 'Failed')),

        CONSTRAINT FK_ml_CaseTrainingRecord_Case
            FOREIGN KEY (IncidentRequestCaseID)
            REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID),

        CONSTRAINT FK_ml_CaseTrainingRecord_ModelVersion
            FOREIGN KEY (EmbeddingModelVersionID)
            REFERENCES ml.EmbeddingModelVersion(EmbeddingModelVersionID)
    );

    PRINT 'ml.CaseTrainingRecord created successfully';
    PRINT '  Unique constraint: IncidentRequestCaseID (one current record per case)';
END
ELSE
BEGIN
    PRINT 'ml.CaseTrainingRecord already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.HistoricalTrainingExample
-- Purpose: Preserve valuable ML training data that cannot reliably be
-- linked to a current operational case (legacy SQLite rows, the orphaned
-- patient_feedback_encoded_Old table, unmatched/conflicting migration
-- rows). No case relationship is required — this is deliberately looser
-- than ml.CaseTrainingRecord because the underlying data has a different
-- integrity guarantee (see ML_ARCHITECTURE_DECISION_RECORD.md section 4.2).
-- All 11 original embedding columns are preserved here even though only 2
-- are actively (re)computed for new operational records.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'HistoricalTrainingExample' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.HistoricalTrainingExample...';

    CREATE TABLE ml.HistoricalTrainingExample (
        HistoricalTrainingExampleID INT IDENTITY(1,1) NOT NULL,

        LegacySource NVARCHAR(100) NULL,
        LegacySourceTable NVARCHAR(200) NULL,
        LegacySourceRowID INT NULL,

        PossibleIncidentRequestCaseID INT NULL,
        LinkConfidence NVARCHAR(20) NULL,

        ComplaintText NVARCHAR(MAX) NULL,
        ImmediateActionText NVARCHAR(MAX) NULL,
        TakenActionText NVARCHAR(MAX) NULL,

        FeedbackTypeID INT NULL,
        DomainID INT NULL,
        CategoryID INT NULL,
        SubCategoryID INT NULL,
        ClassificationID INT NULL,
        SeverityLevelID INT NULL,
        StageID INT NULL,
        HarmLevelID INT NULL,
        ImprovementOpportunityTypeID INT NULL,

        -- All 11 original embedding columns, preserved as authored
        EmbeddingText1 VARBINARY(MAX) NULL,
        EmbeddingText2 VARBINARY(MAX) NULL,
        EmbeddingText3 VARBINARY(MAX) NULL,
        EmbeddingText123 VARBINARY(MAX) NULL,
        EmbeddingText23 VARBINARY(MAX) NULL,
        SentenceEmbedding1 VARBINARY(MAX) NULL,
        SentenceEmbedding2 VARBINARY(MAX) NULL,
        SentenceEmbedding3 VARBINARY(MAX) NULL,
        SentenceEmbedding4 VARBINARY(MAX) NULL,
        SentenceEmbedding5 VARBINARY(MAX) NULL,
        SentenceEmbedding6 VARBINARY(MAX) NULL,

        ImportedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        MigrationBatchID NVARCHAR(100) NULL,
        PreservationNotes NVARCHAR(MAX) NULL,

        CONSTRAINT PK_ml_HistoricalTrainingExample PRIMARY KEY CLUSTERED (HistoricalTrainingExampleID),
        CONSTRAINT CK_ml_HistoricalTrainingExample_LinkConfidence
            CHECK (LinkConfidence IS NULL OR LinkConfidence IN ('Exact', 'High', 'Possible', 'Unmatched', 'Conflict')),

        CONSTRAINT FK_ml_HistoricalTrainingExample_PossibleCase
            FOREIGN KEY (PossibleIncidentRequestCaseID)
            REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID)
    );

    PRINT 'ml.HistoricalTrainingExample created successfully';
    PRINT '  No required case FK — standalone/legacy training assets are first-class here';
END
ELSE
BEGIN
    PRINT 'ml.HistoricalTrainingExample already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.EmbeddingProcessingJob
-- Purpose: Durable, retryable ML processing queue — replaces the current
-- synchronous, unbatched, silently-swallowed-failure embedding calls with
-- an observable, restart-safe job model.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'EmbeddingProcessingJob' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.EmbeddingProcessingJob...';

    CREATE TABLE ml.EmbeddingProcessingJob (
        EmbeddingProcessingJobID INT IDENTITY(1,1) NOT NULL,
        IncidentRequestCaseID INT NOT NULL,

        JobType NVARCHAR(30) NOT NULL,
        Status NVARCHAR(20) NOT NULL DEFAULT 'Pending',

        AttemptCount INT NOT NULL DEFAULT 0,
        MaximumAttempts INT NOT NULL DEFAULT 5,

        RequestedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        StartedAt DATETIME2 NULL,
        CompletedAt DATETIME2 NULL,
        NextRetryAt DATETIME2 NULL,

        LastErrorCode NVARCHAR(50) NULL,
        LastErrorMessage NVARCHAR(MAX) NULL,
        WorkerID NVARCHAR(100) NULL,

        EmbeddingModelVersionID INT NULL,
        ImportBatchID INT NULL,

        CONSTRAINT PK_ml_EmbeddingProcessingJob PRIMARY KEY CLUSTERED (EmbeddingProcessingJobID),
        CONSTRAINT CK_ml_EmbeddingProcessingJob_JobType
            CHECK (JobType IN ('Create', 'Reprocess', 'TextChanged', 'LabelsChanged', 'ModelUpgrade', 'MigrationBackfill')),
        CONSTRAINT CK_ml_EmbeddingProcessingJob_Status
            CHECK (Status IN ('Pending', 'Processing', 'Completed', 'Failed', 'RetryPending', 'Cancelled')),

        CONSTRAINT FK_ml_EmbeddingProcessingJob_Case
            FOREIGN KEY (IncidentRequestCaseID)
            REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID),

        CONSTRAINT FK_ml_EmbeddingProcessingJob_ModelVersion
            FOREIGN KEY (EmbeddingModelVersionID)
            REFERENCES ml.EmbeddingModelVersion(EmbeddingModelVersionID),

        CONSTRAINT FK_ml_EmbeddingProcessingJob_ImportBatch
            FOREIGN KEY (ImportBatchID)
            REFERENCES ml.ImportBatch(ImportBatchID)
    );

    -- Worker needs to efficiently find "give me the next batch of Pending jobs"
    CREATE NONCLUSTERED INDEX IX_ml_EmbeddingProcessingJob_Status_RequestedAt
        ON ml.EmbeddingProcessingJob (Status, RequestedAt);

    PRINT 'ml.EmbeddingProcessingJob created successfully';
END
ELSE
BEGIN
    PRINT 'ml.EmbeddingProcessingJob already exists — skipping';
END
GO

-- ============================================================
-- TABLE: ml.ImportSourceRecordMap
-- Purpose: Record-level import idempotency, generalizing the pattern
-- already proven in dbo.APP_DataMigration_Map (Phase K legacy migration)
-- for the bulk-import pipeline.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'ImportSourceRecordMap' AND SCHEMA_NAME(schema_id) = 'ml')
BEGIN
    PRINT 'Creating TABLE: ml.ImportSourceRecordMap...';

    CREATE TABLE ml.ImportSourceRecordMap (
        ImportSourceRecordMapID INT IDENTITY(1,1) NOT NULL,
        ImportBatchID INT NULL,
        ExternalSourceSystem NVARCHAR(100) NOT NULL,
        ExternalRecordID NVARCHAR(200) NOT NULL,
        IncidentRequestCaseID INT NOT NULL,
        ImportedAt DATETIME2 NOT NULL DEFAULT GETDATE(),

        CONSTRAINT PK_ml_ImportSourceRecordMap PRIMARY KEY CLUSTERED (ImportSourceRecordMapID),
        CONSTRAINT UQ_ml_ImportSourceRecordMap_ExternalRef UNIQUE (ExternalSourceSystem, ExternalRecordID),

        CONSTRAINT FK_ml_ImportSourceRecordMap_Batch
            FOREIGN KEY (ImportBatchID)
            REFERENCES ml.ImportBatch(ImportBatchID),

        CONSTRAINT FK_ml_ImportSourceRecordMap_Case
            FOREIGN KEY (IncidentRequestCaseID)
            REFERENCES dbo.APP_IncidentCase(IncidentRequestCaseID)
    );

    PRINT 'ml.ImportSourceRecordMap created successfully';
    PRINT '  Unique constraint: (ExternalSourceSystem, ExternalRecordID) — catches re-imports';
END
ELSE
BEGIN
    PRINT 'ml.ImportSourceRecordMap already exists — skipping';
END
GO

-- ============================================================
-- TABLE: dbo.SchemaMigrationHistory
-- Purpose: This project currently has no schema-version tracking at all
-- (confirmed during investigation — ~30 ad hoc phase_*.sql files, applied
-- by hand, with nothing recording which have run against a given
-- database). This table starts that tracking, beginning with this very
-- migration.
-- ============================================================
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'SchemaMigrationHistory' AND SCHEMA_NAME(schema_id) = 'dbo')
BEGIN
    PRINT 'Creating TABLE: dbo.SchemaMigrationHistory...';

    CREATE TABLE dbo.SchemaMigrationHistory (
        MigrationID INT IDENTITY(1,1) NOT NULL,
        MigrationName NVARCHAR(255) NOT NULL,
        Checksum NVARCHAR(128) NULL,
        AppliedAt DATETIME2 NOT NULL DEFAULT GETDATE(),
        AppliedBy NVARCHAR(200) NULL,
        ApplicationVersion NVARCHAR(50) NULL,
        Success BIT NOT NULL DEFAULT 1,

        CONSTRAINT PK_dbo_SchemaMigrationHistory PRIMARY KEY CLUSTERED (MigrationID),
        CONSTRAINT UQ_dbo_SchemaMigrationHistory_Name UNIQUE (MigrationName)
    );

    PRINT 'dbo.SchemaMigrationHistory created successfully';
END
ELSE
BEGIN
    PRINT 'dbo.SchemaMigrationHistory already exists — skipping';
END
GO

-- Record this migration itself as the first tracked entry
IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = 'phase_ml_s1_create_ml_schema_and_tables')
BEGIN
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, AppliedBy, Success)
    VALUES ('phase_ml_s1_create_ml_schema_and_tables', SYSTEM_USER, 1);
    PRINT 'Recorded this migration in dbo.SchemaMigrationHistory';
END
GO

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '';
PRINT '============================================================';
PRINT 'PHASE ML-S1 VERIFICATION';
PRINT '============================================================';

SELECT
    s.name AS SchemaName,
    t.name AS TableName,
    p.rows AS ApproxRowCount
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id
JOIN sys.partitions p ON t.object_id = p.object_id AND p.index_id IN (0,1)
WHERE s.name = 'ml' OR t.name = 'SchemaMigrationHistory'
ORDER BY s.name, t.name;

PRINT '';
PRINT 'Phase ML-S1 complete. No existing dbo.* tables were modified.';
PRINT '============================================================';
GO
