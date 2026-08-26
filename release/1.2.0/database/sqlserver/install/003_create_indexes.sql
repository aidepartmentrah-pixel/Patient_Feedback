-- Non-primary-key indexes, generated from live schema.

-- Filtered: IncidentRequestCaseID is NULL for every seasonal-report subcase
-- (see CK_AdministrativeSubcase_ParentLink), and a non-filtered unique index
-- treats all NULLs as duplicates of each other in SQL Server.
IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_APP_AdministrativeSubcase_CaseID')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_APP_AdministrativeSubcase_CaseID] ON [dbo].[APP_AdministrativeSubcase] ([IncidentRequestCaseID] ASC) WHERE [IncidentRequestCaseID] IS NOT NULL;
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_APP_DataMigration_Map_LegacyCase')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_APP_DataMigration_Map_LegacyCase] ON [dbo].[APP_DataMigration_Map] ([legacy_case_id] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_DepartmentEvaluationRule_Department')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_DepartmentEvaluationRule_Department] ON [dbo].[APP_DepartmentEvaluationRule] ([DepartmentID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_DrawerLabel_LabelName')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_DrawerLabel_LabelName] ON [dbo].[APP_DrawerLabel] ([LabelName] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_Hard__4A306784C8A17AC3')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_Hard__4A306784C8A17AC3] ON [dbo].[APP_HardwareConfig] ([ConfigKey] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_APP_IncidentCase_incident_id')
CREATE NONCLUSTERED INDEX [IX_APP_IncidentCase_incident_id] ON [dbo].[APP_IncidentCase] ([incident_id] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ICDoctor')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_ICDoctor] ON [dbo].[APP_IncidentCaseDoctor] ([IncidentRequestCaseID] ASC, [DoctorID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_Employee_Incident')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_Employee_Incident] ON [dbo].[APP_IncidentCaseEmployee] ([EmployeeID] ASC, [IncidentRequestCaseID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_Satisfaction_Case')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_Satisfaction_Case] ON [dbo].[APP_IncidentCaseSatisfaction] ([IncidentRequestCaseID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__D4DA0324BFAB7886')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__D4DA0324BFAB7886] ON [dbo].[APP_LOOKUP_BUILDING] ([BuildingCode] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__8FE31B33F4CFDCC4')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__8FE31B33F4CFDCC4] ON [dbo].[APP_LOOKUP_CASE_STAGE] ([StageName] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__2CC9AC5F79E3204F')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__2CC9AC5F79E3204F] ON [dbo].[APP_LOOKUP_CATEGORY] ([DomainID] ASC, [CategoryName] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_IncidentClassification')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_IncidentClassification] ON [dbo].[APP_LOOKUP_CLASSIFICATION] ([SubCategoryID] ASC, [Classification_AR] ASC, [Classification_EN] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__F432621A6014C23F')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__F432621A6014C23F] ON [dbo].[APP_LOOKUP_DOMAIN] ([DomainCode] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__05E7698AB1140F36')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__05E7698AB1140F36] ON [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ([StatusName] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__83E537E117168E7A')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__83E537E117168E7A] ON [dbo].[APP_LOOKUP_HARM_LEVEL] ([HarmLevel] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_LOOK__A9B2E0025898B253')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_LOOK__A9B2E0025898B253] ON [dbo].[APP_LOOKUP_SUBCATEGORY] ([CategoryID] ASC, [SubCategoryName] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_PublicationBatch_Serial')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_PublicationBatch_Serial] ON [dbo].[APP_PublicationBatch] ([PublicationSerial] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PublicationBatchCase_BatchID')
CREATE NONCLUSTERED INDEX [IX_PublicationBatchCase_BatchID] ON [dbo].[APP_PublicationBatchCase] ([PublicationBatchID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_PublicationBatchCase_IncidentCaseID')
CREATE NONCLUSTERED INDEX [IX_PublicationBatchCase_IncidentCaseID] ON [dbo].[APP_PublicationBatchCase] ([IncidentCaseID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_APP_RCAFactorCategory_Code')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_APP_RCAFactorCategory_Code] ON [dbo].[APP_RCAFactorCategory] ([CategoryCode] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_APP_RCASuggestion_PairedSuggestionID')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_APP_RCASuggestion_PairedSuggestionID] ON [dbo].[APP_RCASuggestion] ([PairedSuggestionID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_Role__D62CB59C33F97644')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_Role__D62CB59C33F97644] ON [dbo].[APP_Roles] ([RoleCode] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ActionItemChangeNotice_Recipient')
CREATE NONCLUSTERED INDEX [IX_ActionItemChangeNotice_Recipient] ON [dbo].[APP_SubcaseActionItemChangeNotice] ([RecipientUserID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UX_ActionItemChangeNotice_PendingPerItem')
CREATE UNIQUE NONCLUSTERED INDEX [UX_ActionItemChangeNotice_PendingPerItem] ON [dbo].[APP_SubcaseActionItemChangeNotice] ([ActionItemID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_SubcaseDecisionAck')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_SubcaseDecisionAck] ON [dbo].[APP_SubcaseDecisionAcknowledgment] ([SubcaseID] ASC, [OrgLevel] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_APP_SubcaseRCASuggestionSelection')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_APP_SubcaseRCASuggestionSelection] ON [dbo].[APP_SubcaseRCASuggestionSelection] ([SubcaseID] ASC, [SuggestionID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SupervisorActionItem_Case')
CREATE NONCLUSTERED INDEX [IX_SupervisorActionItem_Case] ON [dbo].[APP_SupervisorActionItem] ([IncidentRequestCaseID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SupervisorActionItem_Status')
CREATE NONCLUSTERED INDEX [IX_SupervisorActionItem_Status] ON [dbo].[APP_SupervisorActionItem] ([Status] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SupervisorActionItem_TargetOrgUnit')
CREATE NONCLUSTERED INDEX [IX_SupervisorActionItem_TargetOrgUnit] ON [dbo].[APP_SupervisorActionItem] ([TargetOrgUnitID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SupervisorActionItem_TargetUser')
CREATE NONCLUSTERED INDEX [IX_SupervisorActionItem_TargetUser] ON [dbo].[APP_SupervisorActionItem] ([TargetUserID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SupervisorActionItem_Unacknowledged')
CREATE NONCLUSTERED INDEX [IX_SupervisorActionItem_Unacknowledged] ON [dbo].[APP_SupervisorActionItem] ([TargetUserID] ASC, [TargetOrgUnitID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_SupervisorActionItemAuditLog_ActionItem')
CREATE NONCLUSTERED INDEX [IX_SupervisorActionItemAuditLog_ActionItem] ON [dbo].[APP_SupervisorActionItemAuditLog] ([ActionItemID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_Syst__01E719AD73731F68')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_Syst__01E719AD73731F68] ON [dbo].[APP_SystemSettings] ([SettingKey] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_UserRoleScope')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_UserRoleScope] ON [dbo].[APP_UserRoleScope] ([UserID] ASC, [RoleID] ASC, [OrgUnitID] ASC, [OrgUnitType] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ__APP_User__536C85E41C1A652A')
CREATE UNIQUE NONCLUSTERED INDEX [UQ__APP_User__536C85E41C1A652A] ON [dbo].[APP_Users] ([Username] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_dbo_SchemaMigrationHistory_Name')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_dbo_SchemaMigrationHistory_Name] ON [dbo].[SchemaMigrationHistory] ([MigrationName] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ml_CaseTrainingRecord_Case')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_ml_CaseTrainingRecord_Case] ON [ml].[CaseTrainingRecord] ([IncidentRequestCaseID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ml_EmbeddingProcessingJob_Status_RequestedAt')
CREATE NONCLUSTERED INDEX [IX_ml_EmbeddingProcessingJob_Status_RequestedAt] ON [ml].[EmbeddingProcessingJob] ([Status] ASC, [RequestedAt] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ml_HistoricalTrainingExample_LegacySource')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_ml_HistoricalTrainingExample_LegacySource] ON [ml].[HistoricalTrainingExample] ([LegacySourceTable] ASC, [LegacySourceRowID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'IX_ml_ImportBatch_FileChecksum')
CREATE NONCLUSTERED INDEX [IX_ml_ImportBatch_FileChecksum] ON [ml].[ImportBatch] ([FileChecksum] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ml_ImportSourceRecordMap_ExternalRef')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_ml_ImportSourceRecordMap_ExternalRef] ON [ml].[ImportSourceRecordMap] ([ExternalSourceSystem] ASC, [ExternalRecordID] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ml_LegacyDbSizeHistory_RecordDate')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_ml_LegacyDbSizeHistory_RecordDate] ON [ml].[LegacyDbSizeHistory] ([RecordDate] ASC);
GO

IF NOT EXISTS (SELECT 1 FROM sys.indexes WHERE name = 'UQ_ml_LegacyTrainingRunHistory_RunID')
CREATE UNIQUE NONCLUSTERED INDEX [UQ_ml_LegacyTrainingRunHistory_RunID] ON [ml].[LegacyTrainingRunHistory] ([RunID] ASC);
GO
