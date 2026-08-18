-- Rolls back a FRESH INSTALL that has not gone into real use.
-- Do NOT run against a database containing real hospital data -- see README.md.
-- Drops FKs first (any table order is then safe), then all tables this package creates.

-- Step 1: drop foreign keys
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_DataMigration_Map_NewCase')
    ALTER TABLE [dbo].[APP_DataMigration_Map] DROP CONSTRAINT [FK_APP_DataMigration_Map_NewCase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_DataMigration_Map_User')
    ALTER TABLE [dbo].[APP_DataMigration_Map] DROP CONSTRAINT [FK_APP_DataMigration_Map_User];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_IncidentCaseFeedback_IncidentCase')
    ALTER TABLE [dbo].[APP_IncidentCaseFeedback] DROP CONSTRAINT [FK_APP_IncidentCaseFeedback_IncidentCase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_IncidentCase_APP_Incident_incident_id')
    ALTER TABLE [dbo].[APP_IncidentCase] DROP CONSTRAINT [FK_APP_IncidentCase_APP_Incident_incident_id];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_RCASuggestion_Category')
    ALTER TABLE [dbo].[APP_RCASuggestion] DROP CONSTRAINT [FK_APP_RCASuggestion_Category];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_RCASuggestion_PairedSuggestion')
    ALTER TABLE [dbo].[APP_RCASuggestion] DROP CONSTRAINT [FK_APP_RCASuggestion_PairedSuggestion];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_SubcaseRCASuggestionSelection_Subcase')
    ALTER TABLE [dbo].[APP_SubcaseRCASuggestionSelection] DROP CONSTRAINT [FK_APP_SubcaseRCASuggestionSelection_Subcase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_SubcaseRCASuggestionSelection_Suggestion')
    ALTER TABLE [dbo].[APP_SubcaseRCASuggestionSelection] DROP CONSTRAINT [FK_APP_SubcaseRCASuggestionSelection_Suggestion];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_TargetDept_Incident')
    ALTER TABLE [dbo].[APP_IncidentCaseTargetDepartment] DROP CONSTRAINT [FK_APP_TargetDept_Incident];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_AcknowledgedBy')
    ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] DROP CONSTRAINT [FK_ActionItemChangeNotice_AcknowledgedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_ActionItem')
    ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] DROP CONSTRAINT [FK_ActionItemChangeNotice_ActionItem];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_ChangedBy')
    ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] DROP CONSTRAINT [FK_ActionItemChangeNotice_ChangedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_Recipient')
    ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] DROP CONSTRAINT [FK_ActionItemChangeNotice_Recipient];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItem_SeasonalReport')
    ALTER TABLE [dbo].[APP_ActionItem] DROP CONSTRAINT [FK_ActionItem_SeasonalReport];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy')
    ALTER TABLE [dbo].[APP_AdministrativeSubcase] DROP CONSTRAINT [FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy')
    ALTER TABLE [dbo].[APP_AdministrativeSubcase] DROP CONSTRAINT [FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_ForceClosedByUser')
    ALTER TABLE [dbo].[APP_AdministrativeSubcase] DROP CONSTRAINT [FK_AdministrativeSubcase_ForceClosedByUser];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_IncidentCase')
    ALTER TABLE [dbo].[APP_AdministrativeSubcase] DROP CONSTRAINT [FK_AdministrativeSubcase_IncidentCase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_SeasonalReport')
    ALTER TABLE [dbo].[APP_AdministrativeSubcase] DROP CONSTRAINT [FK_AdministrativeSubcase_SeasonalReport];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_SectionExtraTimeGrantedBy')
    ALTER TABLE [dbo].[APP_AdministrativeSubcase] DROP CONSTRAINT [FK_AdministrativeSubcase_SectionExtraTimeGrantedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_DrawerNoteLabelLink_Label')
    ALTER TABLE [dbo].[APP_DrawerNoteLabelLink] DROP CONSTRAINT [FK_DrawerNoteLabelLink_Label];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_DrawerNoteLabelLink_Note')
    ALTER TABLE [dbo].[APP_DrawerNoteLabelLink] DROP CONSTRAINT [FK_DrawerNoteLabelLink_Note];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ICDoctor_Incident')
    ALTER TABLE [dbo].[APP_IncidentCaseDoctor] DROP CONSTRAINT [FK_ICDoctor_Incident];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCaseEmployee_Incident')
    ALTER TABLE [dbo].[APP_IncidentCaseEmployee] DROP CONSTRAINT [FK_IncidentCaseEmployee_Incident];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCaseFeedback_Subcase')
    ALTER TABLE [dbo].[APP_IncidentCaseFeedback] DROP CONSTRAINT [FK_IncidentCaseFeedback_Subcase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCase_ForceClosedByUser')
    ALTER TABLE [dbo].[APP_IncidentCase] DROP CONSTRAINT [FK_IncidentCase_ForceClosedByUser];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCase_RecordType')
    ALTER TABLE [dbo].[APP_IncidentCase] DROP CONSTRAINT [FK_IncidentCase_RecordType];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PolicySnapshot_Report')
    ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport_PolicySnapshot] DROP CONSTRAINT [FK_PolicySnapshot_Report];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PublicationBatchCase_Batch')
    ALTER TABLE [dbo].[APP_PublicationBatchCase] DROP CONSTRAINT [FK_PublicationBatchCase_Batch];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PublicationBatch_PublishedByUser')
    ALTER TABLE [dbo].[APP_PublicationBatch] DROP CONSTRAINT [FK_PublicationBatch_PublishedByUser];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_Satisfaction_IncidentCase')
    ALTER TABLE [dbo].[APP_IncidentCaseSatisfaction] DROP CONSTRAINT [FK_Satisfaction_IncidentCase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_Satisfaction_User')
    ALTER TABLE [dbo].[APP_IncidentCaseSatisfaction] DROP CONSTRAINT [FK_Satisfaction_User];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SeasonalStats_Report')
    ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport_ClassificationStats] DROP CONSTRAINT [FK_SeasonalStats_Report];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SubcaseActionItem_Subcase')
    ALTER TABLE [dbo].[APP_SubcaseActionItem] DROP CONSTRAINT [FK_SubcaseActionItem_Subcase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SubcaseDecisionAck_Subcase')
    ALTER TABLE [dbo].[APP_SubcaseDecisionAcknowledgment] DROP CONSTRAINT [FK_SubcaseDecisionAck_Subcase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItemAuditLog_ActionItem')
    ALTER TABLE [dbo].[APP_SupervisorActionItemAuditLog] DROP CONSTRAINT [FK_SupervisorActionItemAuditLog_ActionItem];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItemAuditLog_PerformedBy')
    ALTER TABLE [dbo].[APP_SupervisorActionItemAuditLog] DROP CONSTRAINT [FK_SupervisorActionItemAuditLog_PerformedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_AcknowledgedBy')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_AcknowledgedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_Case')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_Case];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_CreatedBy')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_CreatedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_Subcase')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_Subcase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_TargetOrgUnit')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_TargetOrgUnit];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_TargetUser')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_TargetUser];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_UpdatedBy')
    ALTER TABLE [dbo].[APP_SupervisorActionItem] DROP CONSTRAINT [FK_SupervisorActionItem_UpdatedBy];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_UserRoleScope_Role')
    ALTER TABLE [dbo].[APP_UserRoleScope] DROP CONSTRAINT [FK_UserRoleScope_Role];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_UserRoleScope_User')
    ALTER TABLE [dbo].[APP_UserRoleScope] DROP CONSTRAINT [FK_UserRoleScope_User];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_CaseTrainingRecord_Case')
    ALTER TABLE [ml].[CaseTrainingRecord] DROP CONSTRAINT [FK_ml_CaseTrainingRecord_Case];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_CaseTrainingRecord_ModelVersion')
    ALTER TABLE [ml].[CaseTrainingRecord] DROP CONSTRAINT [FK_ml_CaseTrainingRecord_ModelVersion];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_EmbeddingProcessingJob_Case')
    ALTER TABLE [ml].[EmbeddingProcessingJob] DROP CONSTRAINT [FK_ml_EmbeddingProcessingJob_Case];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_EmbeddingProcessingJob_ImportBatch')
    ALTER TABLE [ml].[EmbeddingProcessingJob] DROP CONSTRAINT [FK_ml_EmbeddingProcessingJob_ImportBatch];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_EmbeddingProcessingJob_ModelVersion')
    ALTER TABLE [ml].[EmbeddingProcessingJob] DROP CONSTRAINT [FK_ml_EmbeddingProcessingJob_ModelVersion];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_HistoricalTrainingExample_PossibleCase')
    ALTER TABLE [ml].[HistoricalTrainingExample] DROP CONSTRAINT [FK_ml_HistoricalTrainingExample_PossibleCase];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_ImportBatch_User')
    ALTER TABLE [ml].[ImportBatch] DROP CONSTRAINT [FK_ml_ImportBatch_User];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_ImportSourceRecordMap_Batch')
    ALTER TABLE [ml].[ImportSourceRecordMap] DROP CONSTRAINT [FK_ml_ImportSourceRecordMap_Batch];
IF EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_ImportSourceRecordMap_Case')
    ALTER TABLE [ml].[ImportSourceRecordMap] DROP CONSTRAINT [FK_ml_ImportSourceRecordMap_Case];

-- Step 2: drop tables
IF OBJECT_ID('dbo.AdminsrationUnit', 'U') IS NOT NULL
    DROP TABLE [dbo].[AdminsrationUnit];
IF OBJECT_ID('dbo.AdminsrationUnitHistory', 'U') IS NOT NULL
    DROP TABLE [dbo].[AdminsrationUnitHistory];
IF OBJECT_ID('dbo.APP_ActionItem', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_ActionItem];
IF OBJECT_ID('dbo.APP_AdministrativeSubcase', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_AdministrativeSubcase];
IF OBJECT_ID('dbo.APP_CUSTOM_VIEWS', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_CUSTOM_VIEWS];
IF OBJECT_ID('dbo.APP_DataMigration_Map', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_DataMigration_Map];
IF OBJECT_ID('dbo.APP_DepartmentEvaluationRule', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_DepartmentEvaluationRule];
IF OBJECT_ID('dbo.APP_DepartmentPolicy', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_DepartmentPolicy];
IF OBJECT_ID('dbo.APP_DrawerLabel', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_DrawerLabel];
IF OBJECT_ID('dbo.APP_DrawerNote', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_DrawerNote];
IF OBJECT_ID('dbo.APP_DrawerNoteLabelLink', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_DrawerNoteLabelLink];
IF OBJECT_ID('dbo.APP_ExternalApiSettings', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_ExternalApiSettings];
IF OBJECT_ID('dbo.APP_HardwareConfig', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_HardwareConfig];
IF OBJECT_ID('dbo.APP_Incident', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Incident];
IF OBJECT_ID('dbo.APP_IncidentCase', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_IncidentCase];
IF OBJECT_ID('dbo.APP_IncidentCaseDoctor', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_IncidentCaseDoctor];
IF OBJECT_ID('dbo.APP_IncidentCaseEmployee', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_IncidentCaseEmployee];
IF OBJECT_ID('dbo.APP_IncidentCaseFeedback', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_IncidentCaseFeedback];
IF OBJECT_ID('dbo.APP_IncidentCaseSatisfaction', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_IncidentCaseSatisfaction];
IF OBJECT_ID('dbo.APP_IncidentCaseTargetDepartment', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_IncidentCaseTargetDepartment];
IF OBJECT_ID('dbo.APP_LOOKUP_BUILDING', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_BUILDING];
IF OBJECT_ID('dbo.APP_LOOKUP_CASE_STAGE', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_CASE_STAGE];
IF OBJECT_ID('dbo.APP_LOOKUP_CASE_STATUS', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_CASE_STATUS];
IF OBJECT_ID('dbo.APP_LOOKUP_CATEGORY', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_CATEGORY];
IF OBJECT_ID('dbo.APP_LOOKUP_CLASSIFICATION', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_CLASSIFICATION];
IF OBJECT_ID('dbo.APP_LOOKUP_CLINICAL_RISK_TYPE', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE];
IF OBJECT_ID('dbo.APP_LOOKUP_DOCTOR', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_DOCTOR];
IF OBJECT_ID('dbo.APP_LOOKUP_DOMAIN', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_DOMAIN];
IF OBJECT_ID('dbo.APP_LOOKUP_EXPLANATION_STATUS', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_EXPLANATION_STATUS];
IF OBJECT_ID('dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE];
IF OBJECT_ID('dbo.APP_LOOKUP_HARM_LEVEL', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_HARM_LEVEL];
IF OBJECT_ID('dbo.APP_LOOKUP_RECORD_TYPE', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_RECORD_TYPE];
IF OBJECT_ID('dbo.APP_Lookup_SatisfactionStatus', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Lookup_SatisfactionStatus];
IF OBJECT_ID('dbo.APP_LOOKUP_SEVERITY', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_SEVERITY];
IF OBJECT_ID('dbo.APP_LOOKUP_SOURCE', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_SOURCE];
IF OBJECT_ID('dbo.APP_Lookup_SubcaseActionItemStatus', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Lookup_SubcaseActionItemStatus];
IF OBJECT_ID('dbo.APP_Lookup_SubcaseStatus', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Lookup_SubcaseStatus];
IF OBJECT_ID('dbo.APP_Lookup_SubcaseType', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Lookup_SubcaseType];
IF OBJECT_ID('dbo.APP_LOOKUP_SUBCATEGORY', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_LOOKUP_SUBCATEGORY];
IF OBJECT_ID('dbo.APP_OrgUnitPolicy', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_OrgUnitPolicy];
IF OBJECT_ID('dbo.APP_PublicationBatch', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_PublicationBatch];
IF OBJECT_ID('dbo.APP_PublicationBatchCase', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_PublicationBatchCase];
IF OBJECT_ID('dbo.APP_RCAFactorCategory', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_RCAFactorCategory];
IF OBJECT_ID('dbo.APP_RCASuggestion', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_RCASuggestion];
IF OBJECT_ID('dbo.APP_ReportConfig', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_ReportConfig];
IF OBJECT_ID('dbo.APP_RESERVE_DOCTOR', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_RESERVE_DOCTOR];
IF OBJECT_ID('dbo.APP_RESERVE_PATIENT', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_RESERVE_PATIENT];
IF OBJECT_ID('dbo.APP_Roles', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Roles];
IF OBJECT_ID('dbo.APP_SeasonalOrgUnitReport', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SeasonalOrgUnitReport];
IF OBJECT_ID('dbo.APP_SeasonalOrgUnitReport_ClassificationStats', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SeasonalOrgUnitReport_ClassificationStats];
IF OBJECT_ID('dbo.APP_SeasonalOrgUnitReport_PolicySnapshot', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SeasonalOrgUnitReport_PolicySnapshot];
IF OBJECT_ID('dbo.APP_SeasonCase', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SeasonCase];
IF OBJECT_ID('dbo.APP_SubcaseActionItem', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SubcaseActionItem];
IF OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SubcaseActionItemChangeNotice];
IF OBJECT_ID('dbo.APP_SubcaseDecisionAcknowledgment', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SubcaseDecisionAcknowledgment];
IF OBJECT_ID('dbo.APP_SubcaseRCASuggestionSelection', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SubcaseRCASuggestionSelection];
IF OBJECT_ID('dbo.APP_SupervisorActionItem', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SupervisorActionItem];
IF OBJECT_ID('dbo.APP_SupervisorActionItemAuditLog', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SupervisorActionItemAuditLog];
IF OBJECT_ID('dbo.APP_SystemSettings', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_SystemSettings];
IF OBJECT_ID('dbo.APP_UserRoleScope', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_UserRoleScope];
IF OBJECT_ID('dbo.APP_Users', 'U') IS NOT NULL
    DROP TABLE [dbo].[APP_Users];
IF OBJECT_ID('dbo.IncidentRequest', 'U') IS NOT NULL
    DROP TABLE [dbo].[IncidentRequest];
IF OBJECT_ID('dbo.IncidentRequestCase', 'U') IS NOT NULL
    DROP TABLE [dbo].[IncidentRequestCase];
IF OBJECT_ID('dbo.IncidentRequestCaseAction', 'U') IS NOT NULL
    DROP TABLE [dbo].[IncidentRequestCaseAction];
IF OBJECT_ID('dbo.Instance', 'U') IS NOT NULL
    DROP TABLE [dbo].[Instance];
IF OBJECT_ID('dbo.Parameter', 'U') IS NOT NULL
    DROP TABLE [dbo].[Parameter];
IF OBJECT_ID('dbo.Role', 'U') IS NOT NULL
    DROP TABLE [dbo].[Role];
IF OBJECT_ID('dbo.SchemaMigrationHistory', 'U') IS NOT NULL
    DROP TABLE [dbo].[SchemaMigrationHistory];
IF OBJECT_ID('dbo.Season', 'U') IS NOT NULL
    DROP TABLE [dbo].[Season];
IF OBJECT_ID('dbo.UserRole', 'U') IS NOT NULL
    DROP TABLE [dbo].[UserRole];
IF OBJECT_ID('dbo.Users', 'U') IS NOT NULL
    DROP TABLE [dbo].[Users];
IF OBJECT_ID('dbo.VW_HrEmployeeProfileView', 'U') IS NOT NULL
    DROP TABLE [dbo].[VW_HrEmployeeProfileView];
IF OBJECT_ID('ml.CaseTrainingRecord', 'U') IS NOT NULL
    DROP TABLE [ml].[CaseTrainingRecord];
IF OBJECT_ID('ml.EmbeddingModelVersion', 'U') IS NOT NULL
    DROP TABLE [ml].[EmbeddingModelVersion];
IF OBJECT_ID('ml.EmbeddingProcessingJob', 'U') IS NOT NULL
    DROP TABLE [ml].[EmbeddingProcessingJob];
IF OBJECT_ID('ml.HistoricalTrainingExample', 'U') IS NOT NULL
    DROP TABLE [ml].[HistoricalTrainingExample];
IF OBJECT_ID('ml.ImportBatch', 'U') IS NOT NULL
    DROP TABLE [ml].[ImportBatch];
IF OBJECT_ID('ml.ImportSourceRecordMap', 'U') IS NOT NULL
    DROP TABLE [ml].[ImportSourceRecordMap];
IF OBJECT_ID('ml.LegacyDbSizeHistory', 'U') IS NOT NULL
    DROP TABLE [ml].[LegacyDbSizeHistory];
IF OBJECT_ID('ml.LegacyModelMetricHistory', 'U') IS NOT NULL
    DROP TABLE [ml].[LegacyModelMetricHistory];
IF OBJECT_ID('ml.LegacyTrainingRunHistory', 'U') IS NOT NULL
    DROP TABLE [ml].[LegacyTrainingRunHistory];

-- Step 3: drop the ml schema itself if now empty
IF NOT EXISTS (SELECT 1 FROM sys.tables t JOIN sys.schemas s ON t.schema_id = s.schema_id WHERE s.name = 'ml')
    EXEC('DROP SCHEMA ml');