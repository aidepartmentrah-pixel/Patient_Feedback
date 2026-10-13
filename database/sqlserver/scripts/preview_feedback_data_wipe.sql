-- ============================================================================
-- Read-only preview for wipe_feedback_data.sql -- no DELETE, no transaction,
-- nothing is modified. Run this first to see exactly how many rows each
-- table would lose before you run the real script.
-- ============================================================================
SELECT 'dbo.APP_DrawerNoteLabelLink' AS table_name, COUNT(*) AS rows_to_delete FROM dbo.APP_DrawerNoteLabelLink
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
UNION ALL SELECT 'dbo.APP_Incident', COUNT(*) FROM dbo.APP_Incident
ORDER BY table_name;
