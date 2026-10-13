-- ============================================================================
-- Deletes ALL feedback/incident *records* while leaving structure and
-- reference data alone. TAKE A BACKUP FIRST (see backup_database.sql) --
-- this cannot be undone outside a restore.
--
-- v2 -- corrected after a 2026-09-16 run showed two real bugs in v1:
--   1. "WITH CHECK CHECK CONSTRAINT ALL" on the 3 preserved-but-linked
--      tables would have FAILED (and rolled back the whole delete) the
--      moment they held a live reference to a deleted case -- which they
--      always do, that's their purpose. Fixed: those 3 tables' FKs are now
--      re-enabled WITHOUT validation (plain CHECK CONSTRAINT, no WITH
--      CHECK), which is the correct pattern for "these old rows are
--      allowed to point at something that no longer exists."
--   2. v1 reseeded APP_IncidentCase/APP_Incident identities to 0 on every
--      run. That's dangerous even on a clean wipe: APP_DataMigration_Map,
--      ml.HistoricalTrainingExample and ml.ImportSourceRecordMap keep OLD
--      numeric case IDs around on purpose. Reseeding to 0 means a brand
--      new, unrelated case can later be assigned an ID that coincidentally
--      matches one of those old references, which would look like a real
--      link. Fixed: this script never reseeds those two tables. The other
--      21 identity tables aren't referenced by anything outside the wipe,
--      so reseeding them to 0 is safe and still happens.
--
-- Cleared (25 tables): every case/incident/subcase/action-item/drawer-note/
-- publication/seasonal-report row, plus their direct ml.* mirrors
-- (CaseTrainingRecord, EmbeddingProcessingJob).
--
-- Left alone (everything else), on purpose:
--   - APP_LOOKUP_*, APP_Lookup_*, ml.EmbeddingModelVersion  (lookup tables --
--     confirmed against ../validation/validate_lookup_data.sql)
--   - APP_Users, APP_UserRoleScope, APP_UserPasswordExport, APP_Roles
--     (logins -- so people can still sign in after this runs)
--   - APP_RESERVE_DOCTOR / _WORKER / _PATIENT  (Hospital Directory API cache,
--     not feedback data -- re-syncs on its own)
--   - APP_SystemSettings, APP_HardwareConfig, APP_ExternalApiSettings,
--     APP_ReportConfig, APP_CUSTOM_VIEWS, APP_OrgUnitPolicy,
--     APP_DepartmentPolicy, APP_DepartmentEvaluationRule, APP_DrawerLabel,
--     APP_RCAFactorCategory, APP_RCASuggestion  (configuration, not records)
--   - APP_DataMigration_Map, ml.HistoricalTrainingExample,
--     ml.ImportSourceRecordMap, ml.ImportBatch, ml.Legacy*  (migration/import
--     audit trail, not live feedback data -- see note above on why their
--     old case references are deliberately left dangling, not scrubbed)
--   - SchemaMigrationHistory  (never touch -- tracks applied migrations)
--   - The old-system mirror: IncidentRequest, IncidentRequestCase,
--     IncidentRequestCaseAction, Instance, Parameter, Role, Season, UserRole,
--     Users, AdminsrationUnit, AdminsrationUnitHistory,
--     VW_HrEmployeeProfileView (this app doesn't write to these)
--
-- HOW TO RUN IN DBEAVER (a text selection or Ctrl+Enter only runs the
-- statement under the cursor -- that's what caused the partial-execution
-- incident this version fixes):
--   1. Click anywhere in this editor tab, then Ctrl+A to select the WHOLE
--      file.
--   2. Use "Execute SQL Script" (the button with multiple lines / a small
--      script icon, default shortcut Alt+X) -- NOT "Execute SQL Statement"
--      (Ctrl+Enter), which only runs one statement.
--   3. Watch the Output/Execution Log panel (not just the Statistics tab,
--      which only reflects the LAST statement) to confirm every statement
--      ran without error.
-- ============================================================================
SET NOCOUNT ON;
SET XACT_ABORT ON;
-- Some tables here have filtered/unique indexes (e.g. APP_AdministrativeSubcase)
-- that require these ON regardless of whatever a given client's session
-- defaults to -- confirmed needed via direct sqlcmd testing.
SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;

-- Leave this at 1 to preview with zero risk: every DELETE runs for real
-- inside a transaction, before/after counts are shown, then it's rolled
-- back and nothing is kept. Identity reseeding is skipped entirely in dry
-- run (reseed only ever happens after a real, committed delete). Set to 0
-- only once you've reviewed a dry run's output and diagnose_wipe_state.sql.
DECLARE @DryRun bit = 1;
DECLARE @CurrentDb sysname = DB_NAME();

IF @CurrentDb <> N'IncidentManager'
BEGIN
    RAISERROR('This script must be run against the IncidentManager database. Current database is %s. Aborting.', 16, 1, @CurrentDb);
    RETURN;
END

IF @@TRANCOUNT <> 0
BEGIN
    RAISERROR('A transaction is already open in this session (@@TRANCOUNT <> 0). Commit or roll it back before running this script.', 16, 1);
    RETURN;
END

-- Temp tables, not table variables: DBeaver runs the trailing comparison
-- SELECT near the bottom of this script as its own separate call (visible
-- as a distinct ResultSetJobDataRead in its logs), so anything that SELECT
-- reads has to survive across that boundary. Local variables (@before/
-- @after) are scoped to a single batch and would already be gone by then
-- ("Must declare the table variable..."); temp tables (#before/#after) are
-- scoped to the whole session and survive it. Dropped first in case a
-- previous run in this same session left them behind.
IF OBJECT_ID('tempdb..#before') IS NOT NULL DROP TABLE #before;
IF OBJECT_ID('tempdb..#after') IS NOT NULL DROP TABLE #after;
CREATE TABLE #before (table_name sysname, row_count int);
CREATE TABLE #after (table_name sysname, row_count int);

BEGIN TRY
    BEGIN TRANSACTION;

    INSERT INTO #before (table_name, row_count)
    SELECT 'dbo.APP_DrawerNoteLabelLink', COUNT(*) FROM dbo.APP_DrawerNoteLabelLink
    UNION ALL SELECT 'dbo.APP_DrawerNote', COUNT(*) FROM dbo.APP_DrawerNote
    UNION ALL SELECT 'dbo.APP_IncidentCaseDoctor', COUNT(*) FROM dbo.APP_IncidentCaseDoctor
    UNION ALL SELECT 'dbo.APP_IncidentCaseEmployee', COUNT(*) FROM dbo.APP_IncidentCaseEmployee
    UNION ALL SELECT 'dbo.APP_IncidentCaseFeedback', COUNT(*) FROM dbo.APP_IncidentCaseFeedback
    UNION ALL SELECT 'dbo.APP_IncidentCaseSatisfaction', COUNT(*) FROM dbo.APP_IncidentCaseSatisfaction
    UNION ALL SELECT 'dbo.APP_IncidentCaseTargetDepartment', COUNT(*) FROM dbo.APP_IncidentCaseTargetDepartment
    UNION ALL SELECT 'dbo.APP_SubcaseActionItemChangeNotice', COUNT(*) FROM dbo.APP_SubcaseActionItemChangeNotice
    UNION ALL SELECT 'dbo.APP_SubcaseActionItem', COUNT(*) FROM dbo.APP_SubcaseActionItem
    UNION ALL SELECT 'dbo.APP_SubcaseDecisionAcknowledgment', COUNT(*) FROM dbo.APP_SubcaseDecisionAcknowledgment
    UNION ALL SELECT 'dbo.APP_SubcaseRCASuggestionSelection', COUNT(*) FROM dbo.APP_SubcaseRCASuggestionSelection
    UNION ALL SELECT 'dbo.APP_SupervisorActionItemAuditLog', COUNT(*) FROM dbo.APP_SupervisorActionItemAuditLog
    UNION ALL SELECT 'dbo.APP_SupervisorActionItem', COUNT(*) FROM dbo.APP_SupervisorActionItem
    UNION ALL SELECT 'dbo.APP_PublicationBatchCase', COUNT(*) FROM dbo.APP_PublicationBatchCase
    UNION ALL SELECT 'dbo.APP_PublicationBatch', COUNT(*) FROM dbo.APP_PublicationBatch
    UNION ALL SELECT 'dbo.APP_ActionItem', COUNT(*) FROM dbo.APP_ActionItem
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport_ClassificationStats', COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport_PolicySnapshot', COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport', COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport
    UNION ALL SELECT 'dbo.APP_SeasonCase', COUNT(*) FROM dbo.APP_SeasonCase
    UNION ALL SELECT 'dbo.APP_AdministrativeSubcase', COUNT(*) FROM dbo.APP_AdministrativeSubcase
    UNION ALL SELECT 'ml.EmbeddingProcessingJob', COUNT(*) FROM ml.EmbeddingProcessingJob
    UNION ALL SELECT 'ml.CaseTrainingRecord', COUNT(*) FROM ml.CaseTrainingRecord
    UNION ALL SELECT 'dbo.APP_IncidentCase', COUNT(*) FROM dbo.APP_IncidentCase
    UNION ALL SELECT 'dbo.APP_Incident', COUNT(*) FROM dbo.APP_Incident;

    -- Only these 3 constraints are touched, by name -- not "ALL" on any
    -- table, and not on any of the 25 tables being cleared (their delete
    -- order already satisfies every FK on its own, verified against
    -- database/sqlserver/install/004_create_constraints.sql).
    ALTER TABLE dbo.APP_DataMigration_Map NOCHECK CONSTRAINT FK_APP_DataMigration_Map_NewCase;
    ALTER TABLE ml.HistoricalTrainingExample NOCHECK CONSTRAINT FK_ml_HistoricalTrainingExample_PossibleCase;
    ALTER TABLE ml.ImportSourceRecordMap NOCHECK CONSTRAINT FK_ml_ImportSourceRecordMap_Case;

    DELETE FROM dbo.APP_DrawerNoteLabelLink;
    DELETE FROM dbo.APP_DrawerNote;
    DELETE FROM dbo.APP_IncidentCaseDoctor;
    DELETE FROM dbo.APP_IncidentCaseEmployee;
    DELETE FROM dbo.APP_IncidentCaseFeedback;
    DELETE FROM dbo.APP_IncidentCaseSatisfaction;
    DELETE FROM dbo.APP_IncidentCaseTargetDepartment;
    DELETE FROM dbo.APP_SubcaseActionItemChangeNotice;
    DELETE FROM dbo.APP_SubcaseActionItem;
    DELETE FROM dbo.APP_SubcaseDecisionAcknowledgment;
    DELETE FROM dbo.APP_SubcaseRCASuggestionSelection;
    DELETE FROM dbo.APP_SupervisorActionItemAuditLog;
    DELETE FROM dbo.APP_SupervisorActionItem;
    DELETE FROM dbo.APP_PublicationBatchCase;
    DELETE FROM dbo.APP_PublicationBatch;
    DELETE FROM dbo.APP_ActionItem;
    DELETE FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats;
    DELETE FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot;
    DELETE FROM dbo.APP_SeasonalOrgUnitReport;
    DELETE FROM dbo.APP_SeasonCase;
    DELETE FROM dbo.APP_AdministrativeSubcase;
    DELETE FROM ml.EmbeddingProcessingJob;
    DELETE FROM ml.CaseTrainingRecord;
    DELETE FROM dbo.APP_IncidentCase;
    DELETE FROM dbo.APP_Incident;

    -- Re-enable WITHOUT "WITH CHECK": these 3 tables' remaining rows are
    -- expected to reference cases that no longer exist (that's their whole
    -- purpose -- historical/migration audit trail). "WITH CHECK" would
    -- revalidate and fail on exactly those rows. Plain CHECK CONSTRAINT
    -- re-enables enforcement for future inserts/updates without touching
    -- existing rows; sys.foreign_keys.is_not_trusted will show 1 for these
    -- 3 afterwards, which is correct and expected -- not a bug.
    ALTER TABLE dbo.APP_DataMigration_Map CHECK CONSTRAINT FK_APP_DataMigration_Map_NewCase;
    ALTER TABLE ml.HistoricalTrainingExample CHECK CONSTRAINT FK_ml_HistoricalTrainingExample_PossibleCase;
    ALTER TABLE ml.ImportSourceRecordMap CHECK CONSTRAINT FK_ml_ImportSourceRecordMap_Case;

    INSERT INTO #after (table_name, row_count)
    SELECT 'dbo.APP_DrawerNoteLabelLink', COUNT(*) FROM dbo.APP_DrawerNoteLabelLink
    UNION ALL SELECT 'dbo.APP_DrawerNote', COUNT(*) FROM dbo.APP_DrawerNote
    UNION ALL SELECT 'dbo.APP_IncidentCaseDoctor', COUNT(*) FROM dbo.APP_IncidentCaseDoctor
    UNION ALL SELECT 'dbo.APP_IncidentCaseEmployee', COUNT(*) FROM dbo.APP_IncidentCaseEmployee
    UNION ALL SELECT 'dbo.APP_IncidentCaseFeedback', COUNT(*) FROM dbo.APP_IncidentCaseFeedback
    UNION ALL SELECT 'dbo.APP_IncidentCaseSatisfaction', COUNT(*) FROM dbo.APP_IncidentCaseSatisfaction
    UNION ALL SELECT 'dbo.APP_IncidentCaseTargetDepartment', COUNT(*) FROM dbo.APP_IncidentCaseTargetDepartment
    UNION ALL SELECT 'dbo.APP_SubcaseActionItemChangeNotice', COUNT(*) FROM dbo.APP_SubcaseActionItemChangeNotice
    UNION ALL SELECT 'dbo.APP_SubcaseActionItem', COUNT(*) FROM dbo.APP_SubcaseActionItem
    UNION ALL SELECT 'dbo.APP_SubcaseDecisionAcknowledgment', COUNT(*) FROM dbo.APP_SubcaseDecisionAcknowledgment
    UNION ALL SELECT 'dbo.APP_SubcaseRCASuggestionSelection', COUNT(*) FROM dbo.APP_SubcaseRCASuggestionSelection
    UNION ALL SELECT 'dbo.APP_SupervisorActionItemAuditLog', COUNT(*) FROM dbo.APP_SupervisorActionItemAuditLog
    UNION ALL SELECT 'dbo.APP_SupervisorActionItem', COUNT(*) FROM dbo.APP_SupervisorActionItem
    UNION ALL SELECT 'dbo.APP_PublicationBatchCase', COUNT(*) FROM dbo.APP_PublicationBatchCase
    UNION ALL SELECT 'dbo.APP_PublicationBatch', COUNT(*) FROM dbo.APP_PublicationBatch
    UNION ALL SELECT 'dbo.APP_ActionItem', COUNT(*) FROM dbo.APP_ActionItem
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport_ClassificationStats', COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport_PolicySnapshot', COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport', COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport
    UNION ALL SELECT 'dbo.APP_SeasonCase', COUNT(*) FROM dbo.APP_SeasonCase
    UNION ALL SELECT 'dbo.APP_AdministrativeSubcase', COUNT(*) FROM dbo.APP_AdministrativeSubcase
    UNION ALL SELECT 'ml.EmbeddingProcessingJob', COUNT(*) FROM ml.EmbeddingProcessingJob
    UNION ALL SELECT 'ml.CaseTrainingRecord', COUNT(*) FROM ml.CaseTrainingRecord
    UNION ALL SELECT 'dbo.APP_IncidentCase', COUNT(*) FROM dbo.APP_IncidentCase
    UNION ALL SELECT 'dbo.APP_Incident', COUNT(*) FROM dbo.APP_Incident;

    -- Reported HERE, still inside BEGIN TRY, not as a trailing statement
    -- after END CATCH -- DBeaver has been running anything after END CATCH
    -- on what behaves like a separate session (temp tables came back
    -- "Invalid object name" there). Reading it here, before the dry-run
    -- rollback happens, also means a dry run still shows real after_count
    -- values (your own transaction can see its own uncommitted deletes).
    SELECT b.table_name, b.row_count AS before_count, a.row_count AS after_count
    FROM #before b
    JOIN #after a ON a.table_name = b.table_name
    ORDER BY b.table_name;

    IF @DryRun = 1
    BEGIN
        ROLLBACK TRANSACTION;
        PRINT '=== DRY RUN: rolled back. Nothing was actually changed. Counts below show what WOULD happen. ===';
    END
    ELSE
    BEGIN
        -- Reseed only the 21 tables nothing outside this script's clear
        -- list references. APP_IncidentCase and APP_Incident are
        -- deliberately excluded -- see the header note above.
        DBCC CHECKIDENT ('dbo.APP_DrawerNote', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_IncidentCaseDoctor', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_IncidentCaseEmployee', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_IncidentCaseSatisfaction', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_IncidentCaseTargetDepartment', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SubcaseActionItemChangeNotice', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SubcaseActionItem', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SubcaseDecisionAcknowledgment', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SubcaseRCASuggestionSelection', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SupervisorActionItemAuditLog', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SupervisorActionItem', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_PublicationBatchCase', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_PublicationBatch', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_ActionItem', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SeasonalOrgUnitReport_ClassificationStats', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SeasonalOrgUnitReport_PolicySnapshot', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SeasonalOrgUnitReport', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_SeasonCase', RESEED, 0);
        DBCC CHECKIDENT ('dbo.APP_AdministrativeSubcase', RESEED, 0);
        DBCC CHECKIDENT ('ml.EmbeddingProcessingJob', RESEED, 0);
        DBCC CHECKIDENT ('ml.CaseTrainingRecord', RESEED, 0);
        -- dbo.APP_IncidentCase and dbo.APP_Incident: NOT reseeded, on purpose.

        COMMIT TRANSACTION;
        PRINT '=== WIPE COMMITTED. ===';
    END
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION;
    PRINT '=== WIPE FAILED -- transaction rolled back, nothing was changed. ===';
    PRINT 'Error line ' + CAST(ERROR_LINE() AS varchar(10)) + ': ' + ERROR_MESSAGE();
    THROW;
END CATCH

-- Nothing runs after END CATCH on purpose (see note above). #before/#after
-- aren't dropped here either -- the DROP-then-CREATE guard near the top of
-- this script already clears them before the next run.
