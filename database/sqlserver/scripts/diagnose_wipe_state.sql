-- ============================================================================
-- READ-ONLY. Modifies nothing. Run this before touching wipe_feedback_data.sql
-- or repair_identity_reseed.sql, and any time you're unsure what state the
-- database is in after a partial/failed run of either.
--
-- Reports:
--   1. Whether you're connected to the right database, and whether a
--      transaction is currently sitting open in this session.
--   2. Every IDENTITY column touched by the wipe: its current identity
--      value vs. the actual max stored value vs. row count, flagging any
--      table where the next insert would collide with an existing row.
--   3. Every foreign key entering or leaving the tables the wipe touches,
--      including the 3 preserved-but-linked tables, with trusted/disabled
--      status.
--   4. For those 3 preserved tables specifically: how many of their rows
--      currently reference a case that the wipe would delete -- a nonzero
--      count here means "WITH CHECK CHECK CONSTRAINT ALL" would fail if
--      you ever re-add it for these tables.
-- ============================================================================
SET NOCOUNT ON;

PRINT '--- 1. Session state ---';
SELECT DB_NAME() AS current_database,
       CASE WHEN DB_NAME() = N'IncidentManager' THEN 'OK' ELSE 'WRONG DATABASE -- STOP' END AS database_check,
       @@TRANCOUNT AS open_transactions_in_this_session,
       CASE WHEN @@TRANCOUNT = 0 THEN 'OK' ELSE 'A TRANSACTION IS OPEN -- COMMIT OR ROLLBACK BEFORE PROCEEDING' END AS transaction_check;

PRINT '--- 2. Identity health (every identity column the wipe touches) ---';
;WITH id_status (table_name, identity_column, current_identity_value, max_stored_value, row_count) AS (
    SELECT 'dbo.APP_DrawerNote', 'NoteID', IDENT_CURRENT('dbo.APP_DrawerNote'), (SELECT MAX(NoteID) FROM dbo.APP_DrawerNote), (SELECT COUNT(*) FROM dbo.APP_DrawerNote)
    UNION ALL SELECT 'dbo.APP_IncidentCaseDoctor', 'IncidentCaseDoctorID', IDENT_CURRENT('dbo.APP_IncidentCaseDoctor'), (SELECT MAX(IncidentCaseDoctorID) FROM dbo.APP_IncidentCaseDoctor), (SELECT COUNT(*) FROM dbo.APP_IncidentCaseDoctor)
    UNION ALL SELECT 'dbo.APP_IncidentCaseEmployee', 'ID', IDENT_CURRENT('dbo.APP_IncidentCaseEmployee'), (SELECT MAX(ID) FROM dbo.APP_IncidentCaseEmployee), (SELECT COUNT(*) FROM dbo.APP_IncidentCaseEmployee)
    UNION ALL SELECT 'dbo.APP_IncidentCaseSatisfaction', 'SatisfactionID', IDENT_CURRENT('dbo.APP_IncidentCaseSatisfaction'), (SELECT MAX(SatisfactionID) FROM dbo.APP_IncidentCaseSatisfaction), (SELECT COUNT(*) FROM dbo.APP_IncidentCaseSatisfaction)
    UNION ALL SELECT 'dbo.APP_IncidentCaseTargetDepartment', 'TargetID', IDENT_CURRENT('dbo.APP_IncidentCaseTargetDepartment'), (SELECT MAX(TargetID) FROM dbo.APP_IncidentCaseTargetDepartment), (SELECT COUNT(*) FROM dbo.APP_IncidentCaseTargetDepartment)
    UNION ALL SELECT 'dbo.APP_SubcaseActionItemChangeNotice', 'NoticeID', IDENT_CURRENT('dbo.APP_SubcaseActionItemChangeNotice'), (SELECT MAX(NoticeID) FROM dbo.APP_SubcaseActionItemChangeNotice), (SELECT COUNT(*) FROM dbo.APP_SubcaseActionItemChangeNotice)
    UNION ALL SELECT 'dbo.APP_SubcaseActionItem', 'ActionItemID', IDENT_CURRENT('dbo.APP_SubcaseActionItem'), (SELECT MAX(ActionItemID) FROM dbo.APP_SubcaseActionItem), (SELECT COUNT(*) FROM dbo.APP_SubcaseActionItem)
    UNION ALL SELECT 'dbo.APP_SubcaseDecisionAcknowledgment', 'AcknowledgmentID', IDENT_CURRENT('dbo.APP_SubcaseDecisionAcknowledgment'), (SELECT MAX(AcknowledgmentID) FROM dbo.APP_SubcaseDecisionAcknowledgment), (SELECT COUNT(*) FROM dbo.APP_SubcaseDecisionAcknowledgment)
    UNION ALL SELECT 'dbo.APP_SubcaseRCASuggestionSelection', 'SelectionID', IDENT_CURRENT('dbo.APP_SubcaseRCASuggestionSelection'), (SELECT MAX(SelectionID) FROM dbo.APP_SubcaseRCASuggestionSelection), (SELECT COUNT(*) FROM dbo.APP_SubcaseRCASuggestionSelection)
    UNION ALL SELECT 'dbo.APP_SupervisorActionItemAuditLog', 'AuditLogID', IDENT_CURRENT('dbo.APP_SupervisorActionItemAuditLog'), (SELECT MAX(AuditLogID) FROM dbo.APP_SupervisorActionItemAuditLog), (SELECT COUNT(*) FROM dbo.APP_SupervisorActionItemAuditLog)
    UNION ALL SELECT 'dbo.APP_SupervisorActionItem', 'ActionItemID', IDENT_CURRENT('dbo.APP_SupervisorActionItem'), (SELECT MAX(ActionItemID) FROM dbo.APP_SupervisorActionItem), (SELECT COUNT(*) FROM dbo.APP_SupervisorActionItem)
    UNION ALL SELECT 'dbo.APP_PublicationBatchCase', 'PublicationBatchCaseID', IDENT_CURRENT('dbo.APP_PublicationBatchCase'), (SELECT MAX(PublicationBatchCaseID) FROM dbo.APP_PublicationBatchCase), (SELECT COUNT(*) FROM dbo.APP_PublicationBatchCase)
    UNION ALL SELECT 'dbo.APP_PublicationBatch', 'PublicationBatchID', IDENT_CURRENT('dbo.APP_PublicationBatch'), (SELECT MAX(PublicationBatchID) FROM dbo.APP_PublicationBatch), (SELECT COUNT(*) FROM dbo.APP_PublicationBatch)
    UNION ALL SELECT 'dbo.APP_ActionItem', 'ActionItemID', IDENT_CURRENT('dbo.APP_ActionItem'), (SELECT MAX(ActionItemID) FROM dbo.APP_ActionItem), (SELECT COUNT(*) FROM dbo.APP_ActionItem)
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport_ClassificationStats', 'StatID', IDENT_CURRENT('dbo.APP_SeasonalOrgUnitReport_ClassificationStats'), (SELECT MAX(StatID) FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats), (SELECT COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport_ClassificationStats)
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport_PolicySnapshot', 'PolicySnapshotID', IDENT_CURRENT('dbo.APP_SeasonalOrgUnitReport_PolicySnapshot'), (SELECT MAX(PolicySnapshotID) FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot), (SELECT COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport_PolicySnapshot)
    UNION ALL SELECT 'dbo.APP_SeasonalOrgUnitReport', 'SeasonalReportID', IDENT_CURRENT('dbo.APP_SeasonalOrgUnitReport'), (SELECT MAX(SeasonalReportID) FROM dbo.APP_SeasonalOrgUnitReport), (SELECT COUNT(*) FROM dbo.APP_SeasonalOrgUnitReport)
    UNION ALL SELECT 'dbo.APP_SeasonCase', 'SeasonCaseID', IDENT_CURRENT('dbo.APP_SeasonCase'), (SELECT MAX(SeasonCaseID) FROM dbo.APP_SeasonCase), (SELECT COUNT(*) FROM dbo.APP_SeasonCase)
    UNION ALL SELECT 'dbo.APP_AdministrativeSubcase', 'SubcaseID', IDENT_CURRENT('dbo.APP_AdministrativeSubcase'), (SELECT MAX(SubcaseID) FROM dbo.APP_AdministrativeSubcase), (SELECT COUNT(*) FROM dbo.APP_AdministrativeSubcase)
    UNION ALL SELECT 'ml.EmbeddingProcessingJob', 'EmbeddingProcessingJobID', IDENT_CURRENT('ml.EmbeddingProcessingJob'), (SELECT MAX(EmbeddingProcessingJobID) FROM ml.EmbeddingProcessingJob), (SELECT COUNT(*) FROM ml.EmbeddingProcessingJob)
    UNION ALL SELECT 'ml.CaseTrainingRecord', 'CaseTrainingRecordID', IDENT_CURRENT('ml.CaseTrainingRecord'), (SELECT MAX(CaseTrainingRecordID) FROM ml.CaseTrainingRecord), (SELECT COUNT(*) FROM ml.CaseTrainingRecord)
    UNION ALL SELECT 'dbo.APP_IncidentCase', 'IncidentRequestCaseID', IDENT_CURRENT('dbo.APP_IncidentCase'), (SELECT MAX(IncidentRequestCaseID) FROM dbo.APP_IncidentCase), (SELECT COUNT(*) FROM dbo.APP_IncidentCase)
    UNION ALL SELECT 'dbo.APP_Incident', 'incident_id', IDENT_CURRENT('dbo.APP_Incident'), (SELECT MAX(incident_id) FROM dbo.APP_Incident), (SELECT COUNT(*) FROM dbo.APP_Incident)
)
SELECT *,
       CASE WHEN row_count > 0 AND ISNULL(current_identity_value, 0) < ISNULL(max_stored_value, 0)
            THEN 'NEEDS REPAIR -- next insert will collide with an existing row'
            ELSE 'ok' END AS status
FROM id_status
ORDER BY table_name;

PRINT '--- 3. Foreign keys touching any table the wipe clears or preserves-but-links ---';
DECLARE @tables TABLE (name sysname);
INSERT INTO @tables (name) VALUES
 ('APP_DrawerNoteLabelLink'), ('APP_DrawerNote'), ('APP_IncidentCaseDoctor'), ('APP_IncidentCaseEmployee'),
 ('APP_IncidentCaseFeedback'), ('APP_IncidentCaseSatisfaction'), ('APP_IncidentCaseTargetDepartment'),
 ('APP_SubcaseActionItemChangeNotice'), ('APP_SubcaseActionItem'), ('APP_SubcaseDecisionAcknowledgment'),
 ('APP_SubcaseRCASuggestionSelection'), ('APP_SupervisorActionItemAuditLog'), ('APP_SupervisorActionItem'),
 ('APP_PublicationBatchCase'), ('APP_PublicationBatch'), ('APP_ActionItem'),
 ('APP_SeasonalOrgUnitReport_ClassificationStats'), ('APP_SeasonalOrgUnitReport_PolicySnapshot'),
 ('APP_SeasonalOrgUnitReport'), ('APP_SeasonCase'), ('APP_AdministrativeSubcase'),
 ('EmbeddingProcessingJob'), ('CaseTrainingRecord'), ('APP_IncidentCase'), ('APP_Incident'),
 ('APP_DataMigration_Map'), ('HistoricalTrainingExample'), ('ImportSourceRecordMap');

SELECT
    fk.name AS constraint_name,
    OBJECT_SCHEMA_NAME(fk.parent_object_id) + '.' + OBJECT_NAME(fk.parent_object_id) AS referencing_table,
    COL_NAME(fkc.parent_object_id, fkc.parent_column_id) AS referencing_column,
    OBJECT_SCHEMA_NAME(fk.referenced_object_id) + '.' + OBJECT_NAME(fk.referenced_object_id) AS referenced_table,
    COL_NAME(fkc.referenced_object_id, fkc.referenced_column_id) AS referenced_column,
    fk.is_disabled,
    fk.is_not_trusted
FROM sys.foreign_keys fk
JOIN sys.foreign_key_columns fkc ON fkc.constraint_object_id = fk.object_id
WHERE OBJECT_NAME(fk.parent_object_id) IN (SELECT name FROM @tables)
   OR OBJECT_NAME(fk.referenced_object_id) IN (SELECT name FROM @tables)
ORDER BY referencing_table, constraint_name;

PRINT '--- 4. Orphan-risk preview for the 3 preserved-but-linked tables ---';
PRINT '(a nonzero count means WITH CHECK CHECK CONSTRAINT ALL would fail for that table after a real wipe)';
SELECT 'dbo.APP_DataMigration_Map' AS preserved_table,
       COUNT(*) AS rows_referencing_a_case_that_would_be_deleted
FROM dbo.APP_DataMigration_Map m
WHERE EXISTS (SELECT 1 FROM dbo.APP_IncidentCase c WHERE c.IncidentRequestCaseID = m.new_case_id)
UNION ALL
SELECT 'ml.HistoricalTrainingExample', COUNT(*)
FROM ml.HistoricalTrainingExample h
WHERE h.PossibleIncidentRequestCaseID IS NOT NULL
  AND EXISTS (SELECT 1 FROM dbo.APP_IncidentCase c WHERE c.IncidentRequestCaseID = h.PossibleIncidentRequestCaseID)
UNION ALL
SELECT 'ml.ImportSourceRecordMap', COUNT(*)
FROM ml.ImportSourceRecordMap i
WHERE EXISTS (SELECT 1 FROM dbo.APP_IncidentCase c WHERE c.IncidentRequestCaseID = i.IncidentRequestCaseID);
