-- ================================================================
-- PHASE NOTICE: Add RecordTypeID Discriminator Column
-- ================================================================
-- Purpose: Introduce a "Notice" record type alongside "Complaint".
--          Notices reference doctors/workers and appear in profiling
--          and reporting but do NOT require complaint classifications
--          or the RCA/explanation workflow.
--
-- Strategy: Discriminator column on APP_IncidentCase with DEFAULT 1
--           (Complaint). All existing rows auto-migrate to RecordTypeID=1.
--           Classification columns are relaxed to nullable so that
--           Notice rows can legitimately store NULL.
--
-- Date: 2026-05-13
-- ================================================================

USE IncidentManager;
GO

-- ================================================================
-- STEP 1: Create APP_LOOKUP_RECORD_TYPE
-- ================================================================
PRINT 'Step 1: Creating APP_LOOKUP_RECORD_TYPE lookup table...';

IF NOT EXISTS (
    SELECT 1
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'dbo'
      AND TABLE_NAME = 'APP_LOOKUP_RECORD_TYPE'
)
BEGIN
    CREATE TABLE dbo.APP_LOOKUP_RECORD_TYPE (
        RecordTypeID INT          NOT NULL,
        TypeName     NVARCHAR(50) NOT NULL,
        CONSTRAINT PK_APP_LOOKUP_RECORD_TYPE PRIMARY KEY (RecordTypeID)
    );

    INSERT INTO dbo.APP_LOOKUP_RECORD_TYPE (RecordTypeID, TypeName)
    VALUES
        (1, N'Complaint'),
        (2, N'Notice');

    PRINT 'Created APP_LOOKUP_RECORD_TYPE and inserted seed rows.';
END
ELSE
BEGIN
    PRINT 'APP_LOOKUP_RECORD_TYPE already exists, skipping creation.';
END
GO

-- ================================================================
-- STEP 2: Add RecordTypeID discriminator column to APP_IncidentCase
-- ================================================================
PRINT 'Step 2: Adding RecordTypeID column to APP_IncidentCase...';

IF NOT EXISTS (
    SELECT 1
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo'
      AND TABLE_NAME   = 'APP_IncidentCase'
      AND COLUMN_NAME  = 'RecordTypeID'
)
BEGIN
    -- Add with DEFAULT 1 so every existing row becomes RecordTypeID=1 (Complaint)
    ALTER TABLE dbo.APP_IncidentCase
        ADD RecordTypeID INT NOT NULL
        CONSTRAINT DF_APP_IncidentCase_RecordTypeID DEFAULT 1;

    -- Add FK constraint after column exists
    ALTER TABLE dbo.APP_IncidentCase
        ADD CONSTRAINT FK_IncidentCase_RecordType
        FOREIGN KEY (RecordTypeID)
        REFERENCES dbo.APP_LOOKUP_RECORD_TYPE (RecordTypeID);

    PRINT 'RecordTypeID column added with DEFAULT 1 and FK constraint.';
END
ELSE
BEGIN
    PRINT 'RecordTypeID column already exists, skipping.';
END
GO

-- ================================================================
-- STEP 3: Relax classification columns to nullable
-- ================================================================
-- Complaint validation is enforced at the application layer (insert_service.py).
-- These columns are made nullable at the DB level so that Notice rows,
-- which legitimately omit classification data, can be stored without error.
-- Existing complaint rows are unaffected (they already have non-null values).
-- ================================================================
PRINT 'Step 3: Relaxing classification columns to nullable...';

-- DomainID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'DomainID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN DomainID INT NULL;
    PRINT '  DomainID -> NULL';
END

-- CategoryID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'CategoryID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN CategoryID INT NULL;
    PRINT '  CategoryID -> NULL';
END

-- SubCategoryID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'SubCategoryID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN SubCategoryID INT NULL;
    PRINT '  SubCategoryID -> NULL';
END

-- ClassificationID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'ClassificationID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN ClassificationID INT NULL;
    PRINT '  ClassificationID -> NULL';
END

-- SeverityID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'SeverityID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN SeverityID INT NULL;
    PRINT '  SeverityID -> NULL';
END

-- StageID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'StageID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN StageID INT NULL;
    PRINT '  StageID -> NULL';
END

-- HarmLevelID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'HarmLevelID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN HarmLevelID INT NULL;
    PRINT '  HarmLevelID -> NULL';
END

-- ClinicalRiskTypeID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'ClinicalRiskTypeID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN ClinicalRiskTypeID INT NULL;
    PRINT '  ClinicalRiskTypeID -> NULL';
END

-- FeedbackIntentTypeID
IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'FeedbackIntentTypeID' AND IS_NULLABLE = 'NO'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN FeedbackIntentTypeID INT NULL;
    PRINT '  FeedbackIntentTypeID -> NULL';
END

PRINT 'Step 3 complete.';
GO

-- ================================================================
-- STEP 4: Verification
-- ================================================================
PRINT 'Step 4: Running verification queries...';

-- Confirm no rows have NULL RecordTypeID
SELECT
    COUNT(*)                                                       AS TotalRows,
    SUM(CASE WHEN RecordTypeID = 1 THEN 1 ELSE 0 END)             AS Complaints,
    SUM(CASE WHEN RecordTypeID = 2 THEN 1 ELSE 0 END)             AS Notices,
    SUM(CASE WHEN RecordTypeID IS NULL THEN 1 ELSE 0 END)         AS NullRecordType
FROM dbo.APP_IncidentCase;

-- Confirm nullable columns
SELECT
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbo'
  AND TABLE_NAME   = 'APP_IncidentCase'
  AND COLUMN_NAME IN (
      'RecordTypeID', 'DomainID', 'CategoryID', 'SubCategoryID',
      'ClassificationID', 'SeverityID', 'StageID', 'HarmLevelID',
      'ClinicalRiskTypeID', 'FeedbackIntentTypeID'
  )
ORDER BY COLUMN_NAME;

PRINT 'Verification complete.';
GO

-- ================================================================
-- ROLLBACK SCRIPT (for reference only — do not execute)
-- ================================================================
/*
ALTER TABLE dbo.APP_IncidentCase DROP CONSTRAINT FK_IncidentCase_RecordType;
ALTER TABLE dbo.APP_IncidentCase DROP CONSTRAINT DF_APP_IncidentCase_RecordTypeID;
ALTER TABLE dbo.APP_IncidentCase DROP COLUMN RecordTypeID;
DROP TABLE dbo.APP_LOOKUP_RECORD_TYPE;
-- NOTE: Reversing nullable alters is only safe if no Notice rows exist yet.
*/

PRINT '================================================================';
PRINT 'PHASE NOTICE MIGRATION COMPLETED SUCCESSFULLY';
PRINT '================================================================';
GO
