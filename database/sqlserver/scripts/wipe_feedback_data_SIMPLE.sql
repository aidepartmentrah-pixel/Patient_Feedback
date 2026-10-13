-- ============================================================================
-- Deletes ALL feedback/incident records, keeps lookup/config/user/legacy
-- data untouched. Same scope as wipe_feedback_data.sql, rebuilt as a FLAT
-- script: every line below is a complete, independent statement. No
-- DECLARE, no variables, no IF, no BEGIN/END, no TRY/CATCH.
--
-- Why: every failure hit running the earlier (transaction/variable/TRY-CATCH
-- based) version came from something that spanned multiple statements --
-- DBeaver kept splitting those in ways that broke them (a variable declared
-- earlier becomes invisible later, a TRY/CATCH block gets cut mid-way, an
-- IF/BEGIN/END fragments apart). A statement with nothing before or after it
-- to depend on cannot be broken that way, no matter how a client chooses to
-- split the script. This version trades atomicity (no automatic rollback if
-- something fails partway) for reliability -- given nothing currently
-- depends on this data (confirmed with the user 2026-09-17: source of truth
-- is Excel, nothing built on this DB yet), that trade is the right one.
--
-- BEFORE RUNNING: eyeball-confirm you're on the right database and server --
-- run `SELECT DB_NAME(), @@SERVERNAME;` by itself first and read the result.
-- There is no automated check in this script (an automated check would
-- itself be exactly the kind of IF/BEGIN/END block that's been breaking).
--
-- HOW TO RUN IN DBEAVER: click in the editor, Ctrl+A (confirm the highlight
-- covers the very first line to the very last), then "Execute SQL Script"
-- (Alt+X) -- not Ctrl+Enter.
--
-- Cleared (25 tables): every case/incident/subcase/action-item/drawer-note/
-- publication/seasonal-report row, plus their ml.* mirrors.
-- Left alone: APP_LOOKUP_*/APP_Lookup_* (lookup tables), APP_Users and
-- related (logins), APP_RESERVE_* (directory cache), all *Config/*Policy/
-- *Settings tables, APP_DataMigration_Map/ml.HistoricalTrainingExample/
-- ml.ImportSourceRecordMap (migration audit trail -- kept but their FK to
-- the now-deleted cases is intentionally left unvalidated, not scrubbed),
-- SchemaMigrationHistory, and the old-system mirror tables this app never
-- writes to. Full rationale for this split: see wipe_feedback_data.sql's
-- header, which documents the same scope.
-- ============================================================================
-- SET QUOTED_IDENTIFIER/ANSI_NULLS are repeated before EVERY statement
-- below, not just once at the top: a worst-case test (each statement run
-- as its own fresh connection) proved a single SET at the top does not
-- reliably reach a statement 20+ lines later. Repeating it is redundant in
-- the normal case and harmless -- it costs nothing and closes the gap.

SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; ALTER TABLE dbo.APP_DataMigration_Map NOCHECK CONSTRAINT FK_APP_DataMigration_Map_NewCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; ALTER TABLE ml.HistoricalTrainingExample NOCHECK CONSTRAINT FK_ml_HistoricalTrainingExample_PossibleCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; ALTER TABLE ml.ImportSourceRecordMap NOCHECK CONSTRAINT FK_ml_ImportSourceRecordMap_Case;

SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_DrawerNoteLabelLink;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_DrawerNote;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_IncidentCaseDoctor;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_IncidentCaseEmployee;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_IncidentCaseFeedback;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_IncidentCaseSatisfaction;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_IncidentCaseTargetDepartment;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SubcaseActionItemChangeNotice;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SubcaseActionItem;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SubcaseDecisionAcknowledgment;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SubcaseRCASuggestionSelection;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SupervisorActionItemAuditLog;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SupervisorActionItem;
-- AdministrativeSubcase moved here, right after its own dependents and
-- BEFORE anything touching APP_SeasonalOrgUnitReport: it has a live FK
-- (SeasonalReportID -> APP_SeasonalOrgUnitReport.SeasonalReportID) that was
-- previously deleted in the wrong order and blocked the real run on
-- IncidentManager on 2026-09-17. Local test seed data never set that
-- column, which is why this didn't surface in prior validation.
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_AdministrativeSubcase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_PublicationBatchCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_PublicationBatch;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_ActionItem;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SeasonalOrgUnitReport;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_SeasonCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM ml.EmbeddingProcessingJob;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM ml.CaseTrainingRecord;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_IncidentCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; DELETE FROM dbo.APP_Incident;

SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; ALTER TABLE dbo.APP_DataMigration_Map CHECK CONSTRAINT FK_APP_DataMigration_Map_NewCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; ALTER TABLE ml.HistoricalTrainingExample CHECK CONSTRAINT FK_ml_HistoricalTrainingExample_PossibleCase;
SET QUOTED_IDENTIFIER ON; SET ANSI_NULLS ON; ALTER TABLE ml.ImportSourceRecordMap CHECK CONSTRAINT FK_ml_ImportSourceRecordMap_Case;

-- Reseed identities back to 0 so the next inserted row starts at 1.
-- APP_IncidentCase and APP_Incident are NOT reseeded, on purpose: the 3
-- tables above keep old numeric case IDs on file, and reseeding those two
-- would let a brand-new, unrelated case later reuse an old ID and look
-- (wrongly) linked to old migration/history data.
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

-- Verification: every row below should read 0.
SELECT 'dbo.APP_DrawerNoteLabelLink' AS table_name, COUNT(*) AS row_count FROM dbo.APP_DrawerNoteLabelLink
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
