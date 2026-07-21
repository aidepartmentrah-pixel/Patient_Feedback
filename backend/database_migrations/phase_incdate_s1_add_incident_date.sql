-- ============================================================
-- INCDATE-S1: Incident Date field — Database Foundation
-- Date: 2026-07-16
-- Safe to run multiple times (all checks are idempotent)
-- No data deleted. No existing columns altered.
-- ============================================================
--
-- Adds APP_IncidentCase.IncidentDate: the date the incident actually
-- occurred, as distinct from FeedbackRecievedDate (the date the
-- Complaints Office received the complaint).
--
-- Historical rows have no recorded incident date, so per product
-- decision they are backfilled to FeedbackRecievedDate as the closest
-- available approximation. Going forward the column is NOT NULL and
-- the application layer is responsible for collecting a real value.
-- ============================================================

USE IncidentManager;
GO

SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;
GO

PRINT '============================================================';
PRINT 'INCDATE-S1: Incident Date — Database Foundation';
PRINT '============================================================';
PRINT '';

-- ============================================================
-- PART 1: Add IncidentDate column (nullable first, for backfill)
-- ============================================================
PRINT 'PART 1: Adding IncidentDate column to APP_IncidentCase...';

IF NOT EXISTS (SELECT 1 FROM sys.columns WHERE object_id = OBJECT_ID('dbo.APP_IncidentCase') AND name = 'IncidentDate')
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ADD IncidentDate DATE NULL;
    PRINT '  IncidentDate added (nullable)';
END
ELSE
    PRINT '  IncidentDate already exists. Skipping add.';
GO

-- ============================================================
-- PART 2: Backfill existing rows from FeedbackRecievedDate
-- ============================================================
PRINT 'PART 2: Backfilling IncidentDate for existing rows...';

UPDATE dbo.APP_IncidentCase
SET IncidentDate = FeedbackRecievedDate
WHERE IncidentDate IS NULL;

PRINT '  Backfilled ' + CAST(@@ROWCOUNT AS VARCHAR(20)) + ' row(s).';
GO

-- ============================================================
-- PART 3: Enforce NOT NULL going forward
-- ============================================================
PRINT 'PART 3: Enforcing NOT NULL on IncidentDate...';

IF EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_IncidentCase'
      AND COLUMN_NAME = 'IncidentDate' AND IS_NULLABLE = 'YES'
)
BEGIN
    ALTER TABLE dbo.APP_IncidentCase ALTER COLUMN IncidentDate DATE NOT NULL;
    PRINT '  IncidentDate -> NOT NULL';
END
ELSE
    PRINT '  IncidentDate already NOT NULL. Skipping.';
GO

-- ============================================================
-- VERIFICATION
-- ============================================================
PRINT '============================================================';
PRINT 'VERIFICATION';
PRINT '============================================================';

SELECT
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'dbo'
  AND TABLE_NAME   = 'APP_IncidentCase'
  AND COLUMN_NAME IN ('IncidentDate', 'FeedbackRecievedDate', 'CreatedAt', 'UpdatedAt')
ORDER BY COLUMN_NAME;

SELECT
    COUNT(*) AS TotalRows,
    SUM(CASE WHEN IncidentDate IS NULL THEN 1 ELSE 0 END) AS NullIncidentDate,
    SUM(CASE WHEN IncidentDate = FeedbackRecievedDate THEN 1 ELSE 0 END) AS BackfilledFromReceived
FROM dbo.APP_IncidentCase;

PRINT '';
PRINT '============================================================';
PRINT 'INCDATE-S1: Migration complete.';
PRINT '============================================================';
GO

-- ============================================================
-- ROLLBACK SCRIPT (for reference only — do not execute)
-- ============================================================
/*
ALTER TABLE dbo.APP_IncidentCase DROP COLUMN IncidentDate;
*/
