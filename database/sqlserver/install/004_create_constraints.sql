-- Primary key and foreign key constraints, generated from live schema.

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_AdminsrationUnit')
ALTER TABLE [dbo].[AdminsrationUnit] ADD CONSTRAINT [PK_AdminsrationUnit] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_AdminsrationUnitHistory')
ALTER TABLE [dbo].[AdminsrationUnitHistory] ADD CONSTRAINT [PK_AdminsrationUnitHistory] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Acti__56285AD2211B9FB9')
ALTER TABLE [dbo].[APP_ActionItem] ADD CONSTRAINT [PK__APP_Acti__56285AD2211B9FB9] PRIMARY KEY CLUSTERED ([ActionItemID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Admi__32D29BE0AFABD9A3')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [PK__APP_Admi__32D29BE0AFABD9A3] PRIMARY KEY CLUSTERED ([SubcaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_CUST__1E371C16B9CC6387')
ALTER TABLE [dbo].[APP_CUSTOM_VIEWS] ADD CONSTRAINT [PK__APP_CUST__1E371C16B9CC6387] PRIMARY KEY CLUSTERED ([ViewID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_APP_DataMigration_Map')
ALTER TABLE [dbo].[APP_DataMigration_Map] ADD CONSTRAINT [PK_APP_DataMigration_Map] PRIMARY KEY CLUSTERED ([MapID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Depa__D8D6066BE6A1F949')
ALTER TABLE [dbo].[APP_DepartmentEvaluationRule] ADD CONSTRAINT [PK__APP_Depa__D8D6066BE6A1F949] PRIMARY KEY CLUSTERED ([DepartmentEvaluationRuleID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Depa__B2079BCDF18797C3')
ALTER TABLE [dbo].[APP_DepartmentPolicy] ADD CONSTRAINT [PK__APP_Depa__B2079BCDF18797C3] PRIMARY KEY CLUSTERED ([DepartmentID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Draw__397E2BA39E324976')
ALTER TABLE [dbo].[APP_DrawerLabel] ADD CONSTRAINT [PK__APP_Draw__397E2BA39E324976] PRIMARY KEY CLUSTERED ([LabelID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Draw__EACE357FDB56F3D9')
ALTER TABLE [dbo].[APP_DrawerNote] ADD CONSTRAINT [PK__APP_Draw__EACE357FDB56F3D9] PRIMARY KEY CLUSTERED ([NoteID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_DrawerNoteLabelLink')
ALTER TABLE [dbo].[APP_DrawerNoteLabelLink] ADD CONSTRAINT [PK_DrawerNoteLabelLink] PRIMARY KEY CLUSTERED ([LabelID], [NoteID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Exte__C375749DCEA47E9C')
ALTER TABLE [dbo].[APP_ExternalApiSettings] ADD CONSTRAINT [PK__APP_Exte__C375749DCEA47E9C] PRIMARY KEY CLUSTERED ([IntegrationName]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Hard__C3BC333CE50C79EB')
ALTER TABLE [dbo].[APP_HardwareConfig] ADD CONSTRAINT [PK__APP_Hard__C3BC333CE50C79EB] PRIMARY KEY CLUSTERED ([ConfigID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Inci__E6C40DA37ABDD570')
ALTER TABLE [dbo].[APP_Incident] ADD CONSTRAINT [PK__APP_Inci__E6C40DA37ABDD570] PRIMARY KEY CLUSTERED ([incident_id]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Inci__10895CD0C2762BF3')
ALTER TABLE [dbo].[APP_IncidentCase] ADD CONSTRAINT [PK__APP_Inci__10895CD0C2762BF3] PRIMARY KEY CLUSTERED ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Inci__2E1C552A642DB632')
ALTER TABLE [dbo].[APP_IncidentCaseDoctor] ADD CONSTRAINT [PK__APP_Inci__2E1C552A642DB632] PRIMARY KEY CLUSTERED ([IncidentCaseDoctorID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_IncidentCaseEmployee_ID')
ALTER TABLE [dbo].[APP_IncidentCaseEmployee] ADD CONSTRAINT [PK_IncidentCaseEmployee_ID] PRIMARY KEY CLUSTERED ([ID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_APP_IncidentCaseFeedback')
ALTER TABLE [dbo].[APP_IncidentCaseFeedback] ADD CONSTRAINT [PK_APP_IncidentCaseFeedback] PRIMARY KEY CLUSTERED ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Inci__03A8BFAEDD596143')
ALTER TABLE [dbo].[APP_IncidentCaseSatisfaction] ADD CONSTRAINT [PK__APP_Inci__03A8BFAEDD596143] PRIMARY KEY CLUSTERED ([SatisfactionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Inci__2B1F0FB6D15EF7B3')
ALTER TABLE [dbo].[APP_IncidentCaseTargetDepartment] ADD CONSTRAINT [PK__APP_Inci__2B1F0FB6D15EF7B3] PRIMARY KEY CLUSTERED ([TargetID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__5463CDE47B90177D')
ALTER TABLE [dbo].[APP_LOOKUP_BUILDING] ADD CONSTRAINT [PK__APP_LOOK__5463CDE47B90177D] PRIMARY KEY CLUSTERED ([BuildingID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__03EB7AF8AA339216')
ALTER TABLE [dbo].[APP_LOOKUP_CASE_STAGE] ADD CONSTRAINT [PK__APP_LOOK__03EB7AF8AA339216] PRIMARY KEY CLUSTERED ([StageID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__9C43D4D29CF10F99')
ALTER TABLE [dbo].[APP_LOOKUP_CASE_STATUS] ADD CONSTRAINT [PK__APP_LOOK__9C43D4D29CF10F99] PRIMARY KEY CLUSTERED ([CaseStatusID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__19093A2BFF19A802')
ALTER TABLE [dbo].[APP_LOOKUP_CATEGORY] ADD CONSTRAINT [PK__APP_LOOK__19093A2BFF19A802] PRIMARY KEY CLUSTERED ([CategoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__DA747D3145E1BF27')
ALTER TABLE [dbo].[APP_LOOKUP_CLASSIFICATION] ADD CONSTRAINT [PK__APP_LOOK__DA747D3145E1BF27] PRIMARY KEY CLUSTERED ([ClassificationID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__05A6CD55390D6036')
ALTER TABLE [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] ADD CONSTRAINT [PK__APP_LOOK__05A6CD55390D6036] PRIMARY KEY CLUSTERED ([ClinicalRiskTypeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__2DC00EDF70D9D758')
ALTER TABLE [dbo].[APP_LOOKUP_DOCTOR] ADD CONSTRAINT [PK__APP_LOOK__2DC00EDF70D9D758] PRIMARY KEY CLUSTERED ([DoctorID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__2498D77080E7BCB6')
ALTER TABLE [dbo].[APP_LOOKUP_DOMAIN] ADD CONSTRAINT [PK__APP_LOOK__2498D77080E7BCB6] PRIMARY KEY CLUSTERED ([DomainID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__C8EE2043B624CDBA')
ALTER TABLE [dbo].[APP_LOOKUP_EXPLANATION_STATUS] ADD CONSTRAINT [PK__APP_LOOK__C8EE2043B624CDBA] PRIMARY KEY CLUSTERED ([StatusID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__69AF15ED4611AB37')
ALTER TABLE [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] ADD CONSTRAINT [PK__APP_LOOK__69AF15ED4611AB37] PRIMARY KEY CLUSTERED ([FeedbackIntentTypeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__627308877BBBCF07')
ALTER TABLE [dbo].[APP_LOOKUP_HARM_LEVEL] ADD CONSTRAINT [PK__APP_LOOK__627308877BBBCF07] PRIMARY KEY CLUSTERED ([HarmID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_APP_LOOKUP_RECORD_TYPE')
ALTER TABLE [dbo].[APP_LOOKUP_RECORD_TYPE] ADD CONSTRAINT [PK_APP_LOOKUP_RECORD_TYPE] PRIMARY KEY CLUSTERED ([RecordTypeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Look__6109306D11EA945E')
ALTER TABLE [dbo].[APP_Lookup_SatisfactionStatus] ADD CONSTRAINT [PK__APP_Look__6109306D11EA945E] PRIMARY KEY CLUSTERED ([SatisfactionStatusID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__C618A951792A1B3A')
ALTER TABLE [dbo].[APP_LOOKUP_SEVERITY] ADD CONSTRAINT [PK__APP_LOOK__C618A951792A1B3A] PRIMARY KEY CLUSTERED ([SeverityID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__16E019F922D2A908')
ALTER TABLE [dbo].[APP_LOOKUP_SOURCE] ADD CONSTRAINT [PK__APP_LOOK__16E019F922D2A908] PRIMARY KEY CLUSTERED ([SourceID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Look__6A7B44FDCD789CAA')
ALTER TABLE [dbo].[APP_Lookup_SubcaseActionItemStatus] ADD CONSTRAINT [PK__APP_Look__6A7B44FDCD789CAA] PRIMARY KEY CLUSTERED ([StatusCode]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Look__6A7B44FD8D6E81BF')
ALTER TABLE [dbo].[APP_Lookup_SubcaseStatus] ADD CONSTRAINT [PK__APP_Look__6A7B44FD8D6E81BF] PRIMARY KEY CLUSTERED ([StatusCode]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Look__FFBFB493C6F96045')
ALTER TABLE [dbo].[APP_Lookup_SubcaseType] ADD CONSTRAINT [PK__APP_Look__FFBFB493C6F96045] PRIMARY KEY CLUSTERED ([CaseTypeCode]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_LOOK__26BE5BF9B5848A89')
ALTER TABLE [dbo].[APP_LOOKUP_SUBCATEGORY] ADD CONSTRAINT [PK__APP_LOOK__26BE5BF9B5848A89] PRIMARY KEY CLUSTERED ([SubCategoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_APP_OrgUnitPolicy')
ALTER TABLE [dbo].[APP_OrgUnitPolicy] ADD CONSTRAINT [PK_APP_OrgUnitPolicy] PRIMARY KEY CLUSTERED ([OrgUnitPolicyID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Publ__E4EC823963326EDB')
ALTER TABLE [dbo].[APP_PublicationBatch] ADD CONSTRAINT [PK__APP_Publ__E4EC823963326EDB] PRIMARY KEY CLUSTERED ([PublicationBatchID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Publ__2C2B6F478E4FA04C')
ALTER TABLE [dbo].[APP_PublicationBatchCase] ADD CONSTRAINT [PK__APP_Publ__2C2B6F478E4FA04C] PRIMARY KEY CLUSTERED ([PublicationBatchCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_RCAF__19093A2BABAE9FCA')
ALTER TABLE [dbo].[APP_RCAFactorCategory] ADD CONSTRAINT [PK__APP_RCAF__19093A2BABAE9FCA] PRIMARY KEY CLUSTERED ([CategoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_RCAS__9409952885E5330A')
ALTER TABLE [dbo].[APP_RCASuggestion] ADD CONSTRAINT [PK__APP_RCAS__9409952885E5330A] PRIMARY KEY CLUSTERED ([SuggestionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Repo__4A3067856B4693AB')
ALTER TABLE [dbo].[APP_ReportConfig] ADD CONSTRAINT [PK__APP_Repo__4A3067856B4693AB] PRIMARY KEY CLUSTERED ([ConfigKey]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_RESE__2DC00EDF24964E0E')
ALTER TABLE [dbo].[APP_RESERVE_DOCTOR] ADD CONSTRAINT [PK__APP_RESE__2DC00EDF24964E0E] PRIMARY KEY CLUSTERED ([DoctorID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_APP_RESERVE_WORKER')
ALTER TABLE [dbo].[APP_RESERVE_WORKER] ADD CONSTRAINT [PK_APP_RESERVE_WORKER] PRIMARY KEY CLUSTERED ([EmployeeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_RESE__45043C8039AD5214')
ALTER TABLE [dbo].[APP_RESERVE_PATIENT] ADD CONSTRAINT [PK__APP_RESE__45043C8039AD5214] PRIMARY KEY CLUSTERED ([PatientAdmissionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Role__8AFACE3AC7062F84')
ALTER TABLE [dbo].[APP_Roles] ADD CONSTRAINT [PK__APP_Role__8AFACE3AC7062F84] PRIMARY KEY CLUSTERED ([RoleID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_APP_SeasonalOrgUnitReport')
ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport] ADD CONSTRAINT [PK_APP_SeasonalOrgUnitReport] PRIMARY KEY CLUSTERED ([SeasonalReportID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Seas__3A162D1E47732858')
ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport_ClassificationStats] ADD CONSTRAINT [PK__APP_Seas__3A162D1E47732858] PRIMARY KEY CLUSTERED ([StatID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Seas__0A610BB2CC398DD8')
ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport_PolicySnapshot] ADD CONSTRAINT [PK__APP_Seas__0A610BB2CC398DD8] PRIMARY KEY CLUSTERED ([PolicySnapshotID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Seas__72F5A3532295803B')
ALTER TABLE [dbo].[APP_SeasonCase] ADD CONSTRAINT [PK__APP_Seas__72F5A3532295803B] PRIMARY KEY CLUSTERED ([SeasonCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Subc__56285AD28227271C')
ALTER TABLE [dbo].[APP_SubcaseActionItem] ADD CONSTRAINT [PK__APP_Subc__56285AD28227271C] PRIMARY KEY CLUSTERED ([ActionItemID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Subc__CE83CB8539807F2A')
ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] ADD CONSTRAINT [PK__APP_Subc__CE83CB8539807F2A] PRIMARY KEY CLUSTERED ([NoticeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Subc__9342B4AA27CBCF20')
ALTER TABLE [dbo].[APP_SubcaseDecisionAcknowledgment] ADD CONSTRAINT [PK__APP_Subc__9342B4AA27CBCF20] PRIMARY KEY CLUSTERED ([AcknowledgmentID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Subc__7F17912F115C7EA0')
ALTER TABLE [dbo].[APP_SubcaseRCASuggestionSelection] ADD CONSTRAINT [PK__APP_Subc__7F17912F115C7EA0] PRIMARY KEY CLUSTERED ([SelectionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Supe__56285AD2E21DD94D')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [PK__APP_Supe__56285AD2E21DD94D] PRIMARY KEY CLUSTERED ([ActionItemID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Supe__EB5F6CDD9D012BC6')
ALTER TABLE [dbo].[APP_SupervisorActionItemAuditLog] ADD CONSTRAINT [PK__APP_Supe__EB5F6CDD9D012BC6] PRIMARY KEY CLUSTERED ([AuditLogID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_Syst__54372AFDD95AB08C')
ALTER TABLE [dbo].[APP_SystemSettings] ADD CONSTRAINT [PK__APP_Syst__54372AFDD95AB08C] PRIMARY KEY CLUSTERED ([SettingID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_User__9A0B793CDB1F69B1')
ALTER TABLE [dbo].[APP_UserRoleScope] ADD CONSTRAINT [PK__APP_User__9A0B793CDB1F69B1] PRIMARY KEY CLUSTERED ([UserRoleScopeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK__APP_User__1788CCACCB9F4027')
ALTER TABLE [dbo].[APP_Users] ADD CONSTRAINT [PK__APP_User__1788CCACCB9F4027] PRIMARY KEY CLUSTERED ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_UserPasswordExport_User')
ALTER TABLE [dbo].[APP_UserPasswordExport] ADD CONSTRAINT [FK_APP_UserPasswordExport_User] FOREIGN KEY ([UserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_IncidentRequest')
ALTER TABLE [dbo].[IncidentRequest] ADD CONSTRAINT [PK_IncidentRequest] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_IncidentRequestCase')
ALTER TABLE [dbo].[IncidentRequestCase] ADD CONSTRAINT [PK_IncidentRequestCase] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_IncidentRequestCaseAction')
ALTER TABLE [dbo].[IncidentRequestCaseAction] ADD CONSTRAINT [PK_IncidentRequestCaseAction] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_Instance')
ALTER TABLE [dbo].[Instance] ADD CONSTRAINT [PK_Instance] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_Parameter')
ALTER TABLE [dbo].[Parameter] ADD CONSTRAINT [PK_Parameter] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_Role')
ALTER TABLE [dbo].[Role] ADD CONSTRAINT [PK_Role] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_dbo_SchemaMigrationHistory')
ALTER TABLE [dbo].[SchemaMigrationHistory] ADD CONSTRAINT [PK_dbo_SchemaMigrationHistory] PRIMARY KEY CLUSTERED ([MigrationID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_Season')
ALTER TABLE [dbo].[Season] ADD CONSTRAINT [PK_Season] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_UserRole')
ALTER TABLE [dbo].[UserRole] ADD CONSTRAINT [PK_UserRole] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_Users')
ALTER TABLE [dbo].[Users] ADD CONSTRAINT [PK_Users] PRIMARY KEY CLUSTERED ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_CaseTrainingRecord')
ALTER TABLE [ml].[CaseTrainingRecord] ADD CONSTRAINT [PK_ml_CaseTrainingRecord] PRIMARY KEY CLUSTERED ([CaseTrainingRecordID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_EmbeddingModelVersion')
ALTER TABLE [ml].[EmbeddingModelVersion] ADD CONSTRAINT [PK_ml_EmbeddingModelVersion] PRIMARY KEY CLUSTERED ([EmbeddingModelVersionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_EmbeddingProcessingJob')
ALTER TABLE [ml].[EmbeddingProcessingJob] ADD CONSTRAINT [PK_ml_EmbeddingProcessingJob] PRIMARY KEY CLUSTERED ([EmbeddingProcessingJobID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_HistoricalTrainingExample')
ALTER TABLE [ml].[HistoricalTrainingExample] ADD CONSTRAINT [PK_ml_HistoricalTrainingExample] PRIMARY KEY CLUSTERED ([HistoricalTrainingExampleID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_ImportBatch')
ALTER TABLE [ml].[ImportBatch] ADD CONSTRAINT [PK_ml_ImportBatch] PRIMARY KEY CLUSTERED ([ImportBatchID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_ImportSourceRecordMap')
ALTER TABLE [ml].[ImportSourceRecordMap] ADD CONSTRAINT [PK_ml_ImportSourceRecordMap] PRIMARY KEY CLUSTERED ([ImportSourceRecordMapID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_LegacyDbSizeHistory')
ALTER TABLE [ml].[LegacyDbSizeHistory] ADD CONSTRAINT [PK_ml_LegacyDbSizeHistory] PRIMARY KEY CLUSTERED ([LegacyDbSizeHistoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_LegacyModelMetricHistory')
ALTER TABLE [ml].[LegacyModelMetricHistory] ADD CONSTRAINT [PK_ml_LegacyModelMetricHistory] PRIMARY KEY CLUSTERED ([LegacyModelMetricHistoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.key_constraints WHERE name = 'PK_ml_LegacyTrainingRunHistory')
ALTER TABLE [ml].[LegacyTrainingRunHistory] ADD CONSTRAINT [PK_ml_LegacyTrainingRunHistory] PRIMARY KEY CLUSTERED ([LegacyTrainingRunHistoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItem_SeasonalReport')
ALTER TABLE [dbo].[APP_ActionItem] ADD CONSTRAINT [FK_ActionItem_SeasonalReport] FOREIGN KEY ([SeasonalReportID]) REFERENCES [dbo].[APP_SeasonalOrgUnitReport] ([SeasonalReportID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_SeasonalReport')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [FK_AdministrativeSubcase_SeasonalReport] FOREIGN KEY ([SeasonalReportID]) REFERENCES [dbo].[APP_SeasonalOrgUnitReport] ([SeasonalReportID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_ForceClosedByUser')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [FK_AdministrativeSubcase_ForceClosedByUser] FOREIGN KEY ([ForceClosedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_SectionExtraTimeGrantedBy')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [FK_AdministrativeSubcase_SectionExtraTimeGrantedBy] FOREIGN KEY ([SectionExtraTimeGrantedBy]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [FK_AdministrativeSubcase_DepartmentExtraTimeGrantedBy] FOREIGN KEY ([DepartmentExtraTimeGrantedBy]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [FK_AdministrativeSubcase_AdministrationExtraTimeGrantedBy] FOREIGN KEY ([AdministrationExtraTimeGrantedBy]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_AdministrativeSubcase_IncidentCase')
ALTER TABLE [dbo].[APP_AdministrativeSubcase] ADD CONSTRAINT [FK_AdministrativeSubcase_IncidentCase] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_DataMigration_Map_NewCase')
ALTER TABLE [dbo].[APP_DataMigration_Map] ADD CONSTRAINT [FK_APP_DataMigration_Map_NewCase] FOREIGN KEY ([new_case_id]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_DataMigration_Map_User')
ALTER TABLE [dbo].[APP_DataMigration_Map] ADD CONSTRAINT [FK_APP_DataMigration_Map_User] FOREIGN KEY ([migrated_by_user_id]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_DrawerNoteLabelLink_Label')
ALTER TABLE [dbo].[APP_DrawerNoteLabelLink] ADD CONSTRAINT [FK_DrawerNoteLabelLink_Label] FOREIGN KEY ([LabelID]) REFERENCES [dbo].[APP_DrawerLabel] ([LabelID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_DrawerNoteLabelLink_Note')
ALTER TABLE [dbo].[APP_DrawerNoteLabelLink] ADD CONSTRAINT [FK_DrawerNoteLabelLink_Note] FOREIGN KEY ([NoteID]) REFERENCES [dbo].[APP_DrawerNote] ([NoteID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_IncidentCase_APP_Incident_incident_id')
ALTER TABLE [dbo].[APP_IncidentCase] ADD CONSTRAINT [FK_APP_IncidentCase_APP_Incident_incident_id] FOREIGN KEY ([incident_id]) REFERENCES [dbo].[APP_Incident] ([incident_id]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCase_ForceClosedByUser')
ALTER TABLE [dbo].[APP_IncidentCase] ADD CONSTRAINT [FK_IncidentCase_ForceClosedByUser] FOREIGN KEY ([ForceClosedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCase_RecordType')
ALTER TABLE [dbo].[APP_IncidentCase] ADD CONSTRAINT [FK_IncidentCase_RecordType] FOREIGN KEY ([RecordTypeID]) REFERENCES [dbo].[APP_LOOKUP_RECORD_TYPE] ([RecordTypeID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ICDoctor_Incident')
ALTER TABLE [dbo].[APP_IncidentCaseDoctor] ADD CONSTRAINT [FK_ICDoctor_Incident] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCaseEmployee_Incident')
ALTER TABLE [dbo].[APP_IncidentCaseEmployee] ADD CONSTRAINT [FK_IncidentCaseEmployee_Incident] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_IncidentCaseFeedback_IncidentCase')
ALTER TABLE [dbo].[APP_IncidentCaseFeedback] ADD CONSTRAINT [FK_APP_IncidentCaseFeedback_IncidentCase] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_IncidentCaseFeedback_Subcase')
ALTER TABLE [dbo].[APP_IncidentCaseFeedback] ADD CONSTRAINT [FK_IncidentCaseFeedback_Subcase] FOREIGN KEY ([AdministrativeSubcaseID]) REFERENCES [dbo].[APP_AdministrativeSubcase] ([SubcaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_Satisfaction_User')
ALTER TABLE [dbo].[APP_IncidentCaseSatisfaction] ADD CONSTRAINT [FK_Satisfaction_User] FOREIGN KEY ([CreatedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_Satisfaction_IncidentCase')
ALTER TABLE [dbo].[APP_IncidentCaseSatisfaction] ADD CONSTRAINT [FK_Satisfaction_IncidentCase] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_TargetDept_Incident')
ALTER TABLE [dbo].[APP_IncidentCaseTargetDepartment] ADD CONSTRAINT [FK_APP_TargetDept_Incident] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PublicationBatch_PublishedByUser')
ALTER TABLE [dbo].[APP_PublicationBatch] ADD CONSTRAINT [FK_PublicationBatch_PublishedByUser] FOREIGN KEY ([PublishedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PublicationBatchCase_Batch')
ALTER TABLE [dbo].[APP_PublicationBatchCase] ADD CONSTRAINT [FK_PublicationBatchCase_Batch] FOREIGN KEY ([PublicationBatchID]) REFERENCES [dbo].[APP_PublicationBatch] ([PublicationBatchID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_RCASuggestion_Category')
ALTER TABLE [dbo].[APP_RCASuggestion] ADD CONSTRAINT [FK_APP_RCASuggestion_Category] FOREIGN KEY ([CategoryID]) REFERENCES [dbo].[APP_RCAFactorCategory] ([CategoryID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_RCASuggestion_PairedSuggestion')
ALTER TABLE [dbo].[APP_RCASuggestion] ADD CONSTRAINT [FK_APP_RCASuggestion_PairedSuggestion] FOREIGN KEY ([PairedSuggestionID]) REFERENCES [dbo].[APP_RCASuggestion] ([SuggestionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SeasonalStats_Report')
ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport_ClassificationStats] ADD CONSTRAINT [FK_SeasonalStats_Report] FOREIGN KEY ([SeasonalReportID]) REFERENCES [dbo].[APP_SeasonalOrgUnitReport] ([SeasonalReportID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_PolicySnapshot_Report')
ALTER TABLE [dbo].[APP_SeasonalOrgUnitReport_PolicySnapshot] ADD CONSTRAINT [FK_PolicySnapshot_Report] FOREIGN KEY ([SeasonalReportID]) REFERENCES [dbo].[APP_SeasonalOrgUnitReport] ([SeasonalReportID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SubcaseActionItem_Subcase')
ALTER TABLE [dbo].[APP_SubcaseActionItem] ADD CONSTRAINT [FK_SubcaseActionItem_Subcase] FOREIGN KEY ([SubcaseID]) REFERENCES [dbo].[APP_AdministrativeSubcase] ([SubcaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_Recipient')
ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] ADD CONSTRAINT [FK_ActionItemChangeNotice_Recipient] FOREIGN KEY ([RecipientUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_ChangedBy')
ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] ADD CONSTRAINT [FK_ActionItemChangeNotice_ChangedBy] FOREIGN KEY ([ChangedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_AcknowledgedBy')
ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] ADD CONSTRAINT [FK_ActionItemChangeNotice_AcknowledgedBy] FOREIGN KEY ([AcknowledgedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ActionItemChangeNotice_ActionItem')
ALTER TABLE [dbo].[APP_SubcaseActionItemChangeNotice] ADD CONSTRAINT [FK_ActionItemChangeNotice_ActionItem] FOREIGN KEY ([ActionItemID]) REFERENCES [dbo].[APP_SubcaseActionItem] ([ActionItemID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SubcaseDecisionAck_Subcase')
ALTER TABLE [dbo].[APP_SubcaseDecisionAcknowledgment] ADD CONSTRAINT [FK_SubcaseDecisionAck_Subcase] FOREIGN KEY ([SubcaseID]) REFERENCES [dbo].[APP_AdministrativeSubcase] ([SubcaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_SubcaseRCASuggestionSelection_Subcase')
ALTER TABLE [dbo].[APP_SubcaseRCASuggestionSelection] ADD CONSTRAINT [FK_APP_SubcaseRCASuggestionSelection_Subcase] FOREIGN KEY ([SubcaseID]) REFERENCES [dbo].[APP_AdministrativeSubcase] ([SubcaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_APP_SubcaseRCASuggestionSelection_Suggestion')
ALTER TABLE [dbo].[APP_SubcaseRCASuggestionSelection] ADD CONSTRAINT [FK_APP_SubcaseRCASuggestionSelection_Suggestion] FOREIGN KEY ([SuggestionID]) REFERENCES [dbo].[APP_RCASuggestion] ([SuggestionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_AcknowledgedBy')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_AcknowledgedBy] FOREIGN KEY ([AcknowledgedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_Case')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_Case] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_Subcase')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_Subcase] FOREIGN KEY ([SubcaseID]) REFERENCES [dbo].[APP_AdministrativeSubcase] ([SubcaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_TargetOrgUnit')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_TargetOrgUnit] FOREIGN KEY ([TargetOrgUnitID]) REFERENCES [dbo].[AdminsrationUnit] ([UniqueID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_TargetUser')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_TargetUser] FOREIGN KEY ([TargetUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_CreatedBy')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_CreatedBy] FOREIGN KEY ([CreatedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItem_UpdatedBy')
ALTER TABLE [dbo].[APP_SupervisorActionItem] ADD CONSTRAINT [FK_SupervisorActionItem_UpdatedBy] FOREIGN KEY ([UpdatedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItemAuditLog_PerformedBy')
ALTER TABLE [dbo].[APP_SupervisorActionItemAuditLog] ADD CONSTRAINT [FK_SupervisorActionItemAuditLog_PerformedBy] FOREIGN KEY ([PerformedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_SupervisorActionItemAuditLog_ActionItem')
ALTER TABLE [dbo].[APP_SupervisorActionItemAuditLog] ADD CONSTRAINT [FK_SupervisorActionItemAuditLog_ActionItem] FOREIGN KEY ([ActionItemID]) REFERENCES [dbo].[APP_SupervisorActionItem] ([ActionItemID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_UserRoleScope_User')
ALTER TABLE [dbo].[APP_UserRoleScope] ADD CONSTRAINT [FK_UserRoleScope_User] FOREIGN KEY ([UserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_UserRoleScope_Role')
ALTER TABLE [dbo].[APP_UserRoleScope] ADD CONSTRAINT [FK_UserRoleScope_Role] FOREIGN KEY ([RoleID]) REFERENCES [dbo].[APP_Roles] ([RoleID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_CaseTrainingRecord_Case')
ALTER TABLE [ml].[CaseTrainingRecord] ADD CONSTRAINT [FK_ml_CaseTrainingRecord_Case] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_CaseTrainingRecord_ModelVersion')
ALTER TABLE [ml].[CaseTrainingRecord] ADD CONSTRAINT [FK_ml_CaseTrainingRecord_ModelVersion] FOREIGN KEY ([EmbeddingModelVersionID]) REFERENCES [ml].[EmbeddingModelVersion] ([EmbeddingModelVersionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_EmbeddingProcessingJob_ModelVersion')
ALTER TABLE [ml].[EmbeddingProcessingJob] ADD CONSTRAINT [FK_ml_EmbeddingProcessingJob_ModelVersion] FOREIGN KEY ([EmbeddingModelVersionID]) REFERENCES [ml].[EmbeddingModelVersion] ([EmbeddingModelVersionID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_EmbeddingProcessingJob_ImportBatch')
ALTER TABLE [ml].[EmbeddingProcessingJob] ADD CONSTRAINT [FK_ml_EmbeddingProcessingJob_ImportBatch] FOREIGN KEY ([ImportBatchID]) REFERENCES [ml].[ImportBatch] ([ImportBatchID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_EmbeddingProcessingJob_Case')
ALTER TABLE [ml].[EmbeddingProcessingJob] ADD CONSTRAINT [FK_ml_EmbeddingProcessingJob_Case] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_HistoricalTrainingExample_PossibleCase')
ALTER TABLE [ml].[HistoricalTrainingExample] ADD CONSTRAINT [FK_ml_HistoricalTrainingExample_PossibleCase] FOREIGN KEY ([PossibleIncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_ImportBatch_User')
ALTER TABLE [ml].[ImportBatch] ADD CONSTRAINT [FK_ml_ImportBatch_User] FOREIGN KEY ([UploadedByUserID]) REFERENCES [dbo].[APP_Users] ([UserID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_ImportSourceRecordMap_Batch')
ALTER TABLE [ml].[ImportSourceRecordMap] ADD CONSTRAINT [FK_ml_ImportSourceRecordMap_Batch] FOREIGN KEY ([ImportBatchID]) REFERENCES [ml].[ImportBatch] ([ImportBatchID]);
GO

IF NOT EXISTS (SELECT 1 FROM sys.foreign_keys WHERE name = 'FK_ml_ImportSourceRecordMap_Case')
ALTER TABLE [ml].[ImportSourceRecordMap] ADD CONSTRAINT [FK_ml_ImportSourceRecordMap_Case] FOREIGN KEY ([IncidentRequestCaseID]) REFERENCES [dbo].[APP_IncidentCase] ([IncidentRequestCaseID]);
GO
