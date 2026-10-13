-- ============================================================================
-- Undoes the specific damage from a partial wipe run: identity columns that
-- got reseeded to 0 while their tables still hold their original rows (this
-- is what happened to dbo.APP_Incident on 2026-09-16 -- current identity
-- value was reset to 0 while 136 real rows, presumably numbered up to 136,
-- were still present).
--
-- Reseeds every affected identity column back to MAX(its own PK), not to a
-- guessed number -- so it's safe to run even if only some tables were hit.
-- Tables that are genuinely empty get reseeded to 0, which is correct for
-- them.
--
-- SEPARATE from wipe_feedback_data.sql on purpose -- this is a targeted
-- repair for a known-bad state, not something to run routinely. Run
-- diagnose_wipe_state.sql FIRST and read section 2 (identity health) before
-- touching @ConfirmRepair below.
-- ============================================================================
SET NOCOUNT ON;
SET QUOTED_IDENTIFIER ON;
SET ANSI_NULLS ON;

DECLARE @ConfirmRepair bit = 0;  -- change to 1 only after reviewing diagnose_wipe_state.sql section 2
IF @ConfirmRepair = 0
BEGIN
    RAISERROR('Review diagnose_wipe_state.sql output first, then set @ConfirmRepair = 1 in this script before running it.', 16, 1);
    RETURN;
END

DECLARE @CurrentDb sysname = DB_NAME();
IF @CurrentDb <> N'IncidentManager'
BEGIN
    RAISERROR('This script must be run against the IncidentManager database. Current database is %s. Aborting.', 16, 1, @CurrentDb);
    RETURN;
END

BEGIN TRANSACTION;

-- @maxstr holds the reseed value as text, computed BEFORE each EXEC/PRINT --
-- EXEC(...)'s dynamic-SQL-string argument only accepts string literals and
-- variables concatenated with '+', not an inline CAST(...) call (confirmed
-- via direct testing: "Incorrect syntax near 'CAST'" on every one of these
-- when CAST was inline).
DECLARE @max int, @maxstr varchar(20);

SELECT @max = MAX(NoteID) FROM dbo.APP_DrawerNote;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_DrawerNote'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_DrawerNote -> reseeded to ' + @maxstr;

SELECT @max = MAX(IncidentCaseDoctorID) FROM dbo.APP_IncidentCaseDoctor;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_IncidentCaseDoctor'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_IncidentCaseDoctor -> reseeded to ' + @maxstr;

SELECT @max = MAX(ID) FROM dbo.APP_IncidentCaseEmployee;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_IncidentCaseEmployee'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_IncidentCaseEmployee -> reseeded to ' + @maxstr;

SELECT @max = MAX(SatisfactionID) FROM dbo.APP_IncidentCaseSatisfaction;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_IncidentCaseSatisfaction'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_IncidentCaseSatisfaction -> reseeded to ' + @maxstr;

SELECT @max = MAX(TargetID) FROM dbo.APP_IncidentCaseTargetDepartment;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_IncidentCaseTargetDepartment'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_IncidentCaseTargetDepartment -> reseeded to ' + @maxstr;

SELECT @max = MAX(NoticeID) FROM dbo.APP_SubcaseActionItemChangeNotice;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SubcaseActionItemChangeNotice'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SubcaseActionItemChangeNotice -> reseeded to ' + @maxstr;

SELECT @max = MAX(ActionItemID) FROM dbo.APP_SubcaseActionItem;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SubcaseActionItem'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SubcaseActionItem -> reseeded to ' + @maxstr;

SELECT @max = MAX(AcknowledgmentID) FROM dbo.APP_SubcaseDecisionAcknowledgment;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SubcaseDecisionAcknowledgment'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SubcaseDecisionAcknowledgment -> reseeded to ' + @maxstr;

SELECT @max = MAX(SelectionID) FROM dbo.APP_SubcaseRCASuggestionSelection;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SubcaseRCASuggestionSelection'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SubcaseRCASuggestionSelection -> reseeded to ' + @maxstr;

SELECT @max = MAX(AuditLogID) FROM dbo.APP_SupervisorActionItemAuditLog;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SupervisorActionItemAuditLog'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SupervisorActionItemAuditLog -> reseeded to ' + @maxstr;

SELECT @max = MAX(ActionItemID) FROM dbo.APP_SupervisorActionItem;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SupervisorActionItem'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SupervisorActionItem -> reseeded to ' + @maxstr;

SELECT @max = MAX(PublicationBatchCaseID) FROM dbo.APP_PublicationBatchCase;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_PublicationBatchCase'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_PublicationBatchCase -> reseeded to ' + @maxstr;

SELECT @max = MAX(PublicationBatchID) FROM dbo.APP_PublicationBatch;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_PublicationBatch'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_PublicationBatch -> reseeded to ' + @maxstr;

SELECT @max = MAX(ActionItemID) FROM dbo.APP_ActionItem;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_ActionItem'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_ActionItem -> reseeded to ' + @maxstr;

SELECT @max = MAX(StatID) FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SeasonalOrgUnitReport_ClassificationStats'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SeasonalOrgUnitReport_ClassificationStats -> reseeded to ' + @maxstr;

SELECT @max = MAX(PolicySnapshotID) FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SeasonalOrgUnitReport_PolicySnapshot'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SeasonalOrgUnitReport_PolicySnapshot -> reseeded to ' + @maxstr;

SELECT @max = MAX(SeasonalReportID) FROM dbo.APP_SeasonalOrgUnitReport;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SeasonalOrgUnitReport'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SeasonalOrgUnitReport -> reseeded to ' + @maxstr;

SELECT @max = MAX(SeasonCaseID) FROM dbo.APP_SeasonCase;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_SeasonCase'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_SeasonCase -> reseeded to ' + @maxstr;

SELECT @max = MAX(SubcaseID) FROM dbo.APP_AdministrativeSubcase;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_AdministrativeSubcase'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_AdministrativeSubcase -> reseeded to ' + @maxstr;

SELECT @max = MAX(EmbeddingProcessingJobID) FROM ml.EmbeddingProcessingJob;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''ml.EmbeddingProcessingJob'', RESEED, ' + @maxstr + ')');
PRINT 'ml.EmbeddingProcessingJob -> reseeded to ' + @maxstr;

SELECT @max = MAX(CaseTrainingRecordID) FROM ml.CaseTrainingRecord;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''ml.CaseTrainingRecord'', RESEED, ' + @maxstr + ')');
PRINT 'ml.CaseTrainingRecord -> reseeded to ' + @maxstr;

SELECT @max = MAX(IncidentRequestCaseID) FROM dbo.APP_IncidentCase;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_IncidentCase'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_IncidentCase -> reseeded to ' + @maxstr;

SELECT @max = MAX(incident_id) FROM dbo.APP_Incident;
SET @maxstr = CAST(ISNULL(@max, 0) AS varchar(20));
EXEC('DBCC CHECKIDENT (''dbo.APP_Incident'', RESEED, ' + @maxstr + ')');
PRINT 'dbo.APP_Incident -> reseeded to ' + @maxstr;

COMMIT TRANSACTION;
PRINT '=== Identity repair committed. Re-run diagnose_wipe_state.sql section 2 to confirm every row now says "ok". ===';
