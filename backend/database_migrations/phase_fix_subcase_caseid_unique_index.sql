-- ============================================================
-- FIX: UQ_APP_AdministrativeSubcase_CaseID must ignore NULLs
-- ============================================================
-- Purpose: The unique index on IncidentRequestCaseID was created without
-- a WHERE filter, so SQL Server treats every NULL IncidentRequestCaseID
-- (i.e. every seasonal-report subcase, which has no incident and attaches
-- to a SeasonalReportID instead -- see CK_AdministrativeSubcase_ParentLink)
-- as a duplicate of every other NULL. Only the first seasonal-report
-- subcase ever inserted succeeds; every subsequent one hits error 2601
-- and silently fails to be created (caught and logged as a warning in
-- seasonal_report_generator.py, so report generation "succeeds" but the
-- workflow subcase never does).
--
-- Fix: recreate the index as filtered, matching the pattern already used
-- by its sibling indexes IX_AdministrativeSubcase_Incident and
-- IX_AdministrativeSubcase_Seasonal in
-- phase3_step2_create_administrative_subcase.sql.
--
-- Safety: Drops and recreates ONE existing index. No data loss -- an
-- index carries no data of its own, just an ordering/lookup structure.
-- Idempotent: safe to run multiple times.
-- ============================================================

USE IncidentManager;
GO

PRINT '============================================================';
PRINT 'FIX: UQ_APP_AdministrativeSubcase_CaseID -- filter out NULLs';
PRINT '============================================================';

IF EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'UQ_APP_AdministrativeSubcase_CaseID'
    AND object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
    AND has_filter = 0
)
BEGIN
    DROP INDEX [UQ_APP_AdministrativeSubcase_CaseID] ON dbo.APP_AdministrativeSubcase;
    PRINT '  Dropped old non-filtered UQ_APP_AdministrativeSubcase_CaseID.';
END
ELSE
BEGIN
    PRINT '  No non-filtered UQ_APP_AdministrativeSubcase_CaseID found (already fixed or not yet created).';
END

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE name = 'UQ_APP_AdministrativeSubcase_CaseID'
    AND object_id = OBJECT_ID('dbo.APP_AdministrativeSubcase')
)
BEGIN
    CREATE UNIQUE NONCLUSTERED INDEX [UQ_APP_AdministrativeSubcase_CaseID]
    ON dbo.APP_AdministrativeSubcase ([IncidentRequestCaseID] ASC)
    WHERE [IncidentRequestCaseID] IS NOT NULL;

    PRINT '  Created filtered UQ_APP_AdministrativeSubcase_CaseID (excludes NULLs).';
END
ELSE
BEGIN
    PRINT '  UQ_APP_AdministrativeSubcase_CaseID already filtered. Skipping.';
END
GO

PRINT '';
PRINT '✅ UQ_APP_AdministrativeSubcase_CaseID fix complete.';
PRINT '';
