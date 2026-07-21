-- Auto-generated from live IncidentManager schema inspection (read-only). No business data included.
-- Obsolete objects (VW_PatientAdmission, VW_Doctors) are intentionally NOT created on fresh installs;
-- see docs/DATABASE_STRUCTURE_REPORT.md and database/sqlserver/retirement/.
SET ANSI_NULLS ON;
SET QUOTED_IDENTIFIER ON;
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'ml')
    EXEC('CREATE SCHEMA ml');
GO

IF OBJECT_ID('dbo.AdminsrationUnit', 'U') IS NULL
CREATE TABLE [dbo].[AdminsrationUnit] (
    [UniqueID] int NOT NULL,
    [Name] nvarchar(50) NULL,
    [ParentID] int NULL,
    [Frozen] bit NULL,
    [Type] int NULL,
    [CreateDate] datetime NULL,
    [CreateID] int NULL,
    [UpdateDate] datetime NULL,
    [UpdateUser] int NULL
);
GO

IF OBJECT_ID('dbo.AdminsrationUnitHistory', 'U') IS NULL
CREATE TABLE [dbo].[AdminsrationUnitHistory] (
    [UniqueID] int IDENTITY(1,1) NOT NULL,
    [AdminID] int NULL,
    [Name] nvarchar(50) NULL,
    [ParentID] int NULL,
    [Frozen] bit NULL,
    [CreateDate] datetime NULL,
    [CreateID] int NULL,
    [UpdateDate] datetime NULL,
    [UpdateUser] int NULL
);
GO

IF OBJECT_ID('dbo.APP_ActionItem', 'U') IS NULL
CREATE TABLE [dbo].[APP_ActionItem] (
    [ActionItemID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NULL,
    [SeasonCaseID] int NULL,
    [ActionTitle] nvarchar(300) NOT NULL,
    [ActionDescription] nvarchar(MAX) NULL,
    [DueDate] date NULL,
    [DateSubmitted] date NULL,
    [IsDone] bit NOT NULL DEFAULT ((0)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL,
    [SeasonalReportID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_AdministrativeSubcase', 'U') IS NULL
CREATE TABLE [dbo].[APP_AdministrativeSubcase] (
    [SubcaseID] int IDENTITY(1,1) NOT NULL,
    [CaseType] nvarchar(50) NOT NULL,
    [IncidentRequestCaseID] int NULL,
    [SeasonalReportID] int NULL,
    [TargetOrgUnitID] int NOT NULL,
    [Status] nvarchar(50) NOT NULL,
    [SectionExplanationText] nvarchar(MAX) NULL,
    [SectionRejectionText] nvarchar(MAX) NULL,
    [DepartmentExplanationText] nvarchar(MAX) NULL,
    [DepartmentRejectionText] nvarchar(MAX) NULL,
    [AdministrationExplanationText] nvarchar(MAX) NULL,
    [AdministrationRejectionText] nvarchar(MAX) NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL,
    [UpdatedAt] datetime NULL,
    [UpdatedByUserID] int NULL,
    [ForceClosedAt] datetime NULL,
    [ForceClosedByUserID] int NULL,
    [ForceCloseReason] nvarchar(MAX) NULL,
    [SectionEnteredByUserID] int NULL,
    [SectionEnteredForRole] nvarchar(50) NULL,
    [SectionEntryMode] nvarchar(50) NULL,
    [SectionEntryTimestamp] datetime NULL,
    [DepartmentEnteredByUserID] int NULL,
    [DepartmentEnteredForRole] nvarchar(50) NULL,
    [DepartmentEntryMode] nvarchar(50) NULL,
    [DepartmentEntryTimestamp] datetime NULL,
    [AdministrationEnteredByUserID] int NULL,
    [AdministrationEnteredForRole] nvarchar(50) NULL,
    [AdministrationEntryMode] nvarchar(50) NULL,
    [AdministrationEntryTimestamp] datetime NULL,
    [PatientServicesDecisionText] nvarchar(MAX) NULL,
    [PatientServicesDecisionByUserID] int NULL,
    [PatientServicesDecisionAt] datetime NULL,
    [PatientServicesDecisionUpdatedAt] datetime NULL,
    [SectionDeadlineAt] datetime NULL,
    [DepartmentDeadlineAt] datetime NULL,
    [AdministrationDeadlineAt] datetime NULL,
    [SectionForceClosedAt] datetime NULL,
    [DepartmentForceClosedAt] datetime NULL,
    [AdministrationForceClosedAt] datetime NULL,
    [SectionLateReply] bit NOT NULL DEFAULT ((0)),
    [DepartmentLateReply] bit NOT NULL DEFAULT ((0)),
    [AdministrationLateReply] bit NOT NULL DEFAULT ((0)),
    [SectionExtraTimeGrantedAt] datetime NULL,
    [DepartmentExtraTimeGrantedAt] datetime NULL,
    [AdministrationExtraTimeGrantedAt] datetime NULL,
    [SectionExtraTimeGrantedBy] int NULL,
    [DepartmentExtraTimeGrantedBy] int NULL,
    [AdministrationExtraTimeGrantedBy] int NULL
);
GO

IF OBJECT_ID('dbo.APP_CUSTOM_VIEWS', 'U') IS NULL
CREATE TABLE [dbo].[APP_CUSTOM_VIEWS] (
    [ViewID] int IDENTITY(1,1) NOT NULL,
    [ViewName] nvarchar(150) NOT NULL,
    [ShowIncidentRequestCaseID] bit NOT NULL DEFAULT ((1)),
    [ShowComplaintText] bit NOT NULL DEFAULT ((1)),
    [ShowImmediateAction] bit NOT NULL DEFAULT ((0)),
    [ShowTakenAction] bit NOT NULL DEFAULT ((0)),
    [ShowFeedbackRecievedDate] bit NOT NULL DEFAULT ((1)),
    [ShowPatientName] bit NOT NULL DEFAULT ((1)),
    [ShowIssuingOrgUnitID] bit NOT NULL DEFAULT ((0)),
    [ShowCreatedAt] bit NOT NULL DEFAULT ((1)),
    [ShowCreatedByUserID] bit NOT NULL DEFAULT ((0)),
    [ShowIsInPatient] bit NOT NULL DEFAULT ((0)),
    [ShowClinicalRiskTypeID] bit NOT NULL DEFAULT ((0)),
    [ShowFeedbackIntentTypeID] bit NOT NULL DEFAULT ((0)),
    [ShowBuildingID] bit NOT NULL DEFAULT ((0)),
    [ShowDomainID] bit NOT NULL DEFAULT ((1)),
    [ShowCategoryID] bit NOT NULL DEFAULT ((1)),
    [ShowSubCategoryID] bit NOT NULL DEFAULT ((1)),
    [ShowClassificationID] bit NOT NULL DEFAULT ((1)),
    [ShowSeverityID] bit NOT NULL DEFAULT ((1)),
    [ShowStageID] bit NOT NULL DEFAULT ((1)),
    [ShowHarmLevelID] bit NOT NULL DEFAULT ((0)),
    [ShowCaseStatusID] bit NOT NULL DEFAULT ((1)),
    [ShowSourceID] bit NOT NULL DEFAULT ((0)),
    [ShowExplanationStatusID] bit NOT NULL DEFAULT ((0)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [ShowIncidentNumber] bit NOT NULL DEFAULT ((0)),
    [ShowSectionAnswer] bit NOT NULL DEFAULT ((0)),
    [ShowDepartmentAnswer] bit NOT NULL DEFAULT ((0)),
    [ShowAdministrationAnswer] bit NOT NULL DEFAULT ((0)),
    [ShowTargetDepartment] bit NOT NULL DEFAULT ((0)),
    [ShowSatisfactionStatus] bit NOT NULL DEFAULT ((0)),
    [ShowSatisfactionDate] bit NOT NULL DEFAULT ((0)),
    [ShowRedFlagIndicator] bit NOT NULL DEFAULT ((0)),
    [ShowNeverEventIndicator] bit NOT NULL DEFAULT ((0)),
    [ShowMorbidityIndicator] bit NOT NULL DEFAULT ((0)),
    [ShowLateIndicator] bit NOT NULL DEFAULT ((0)),
    [ShowForceClosedIndicator] bit NOT NULL DEFAULT ((0)),
    [ShowLastEdited] bit NOT NULL DEFAULT ((0)),
    [ShowIncidentDate] bit NOT NULL DEFAULT ((0)),
    [ShowPublicationDate] bit NOT NULL DEFAULT ((0)),
    [ShowRcaReplies] bit NOT NULL DEFAULT ((0)),
    [ShowComplaintSummary] bit NOT NULL DEFAULT ((0)),
    [ShowCustomerServiceDecision] bit NOT NULL DEFAULT ((0)),
    [ShowCustomerServiceDecisionDate] bit NOT NULL DEFAULT ((0)),
    [ShowRecordType] bit NOT NULL DEFAULT ((0)),
    [ShowSectionEntry] bit NOT NULL DEFAULT ((0)),
    [ShowSectionDeadline] bit NOT NULL DEFAULT ((0)),
    [ShowDepartmentEntry] bit NOT NULL DEFAULT ((0)),
    [ShowDepartmentDeadline] bit NOT NULL DEFAULT ((0)),
    [ShowAdministrationEntry] bit NOT NULL DEFAULT ((0)),
    [ShowAdministrationDeadline] bit NOT NULL DEFAULT ((0))
);
GO

IF OBJECT_ID('dbo.APP_DataMigration_Map', 'U') IS NULL
CREATE TABLE [dbo].[APP_DataMigration_Map] (
    [MapID] int IDENTITY(1,1) NOT NULL,
    [legacy_case_id] int NOT NULL,
    [new_case_id] int NOT NULL,
    [migrated_by_user_id] int NOT NULL,
    [migrated_at] datetime2 NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_DepartmentEvaluationRule', 'U') IS NULL
CREATE TABLE [dbo].[APP_DepartmentEvaluationRule] (
    [DepartmentEvaluationRuleID] int IDENTITY(1,1) NOT NULL,
    [DepartmentID] int NOT NULL,
    [LowSeverityMaxCount] int NULL,
    [MediumSeverityMaxCount] int NULL,
    [HighSeverityMaxCount] int NULL,
    [HighSeverityMaxPercentage] decimal(5,2) NULL,
    [HighSeverityByDomainMaxPercentage] decimal(5,2) NULL,
    [EnableLowSeverityRepetitionRule] bit NOT NULL DEFAULT ((0)),
    [EnableMediumSeverityRepetitionRule] bit NOT NULL DEFAULT ((0)),
    [EnableHighSeverityPercentageRule] bit NOT NULL DEFAULT ((0)),
    [EnableHighSeverityPercentageByDomainRule] bit NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_DepartmentPolicy', 'U') IS NULL
CREATE TABLE [dbo].[APP_DepartmentPolicy] (
    [DepartmentID] int NOT NULL,
    [PolicyData] nvarchar(MAX) NULL,
    [CreatedAt] datetime NULL DEFAULT (getdate()),
    [UpdatedAt] datetime NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_DrawerLabel', 'U') IS NULL
CREATE TABLE [dbo].[APP_DrawerLabel] (
    [LabelID] int IDENTITY(1,1) NOT NULL,
    [LabelName] nvarchar(100) NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime2 NOT NULL DEFAULT (sysutcdatetime())
);
GO

IF OBJECT_ID('dbo.APP_DrawerNote', 'U') IS NULL
CREATE TABLE [dbo].[APP_DrawerNote] (
    [NoteID] int IDENTITY(1,1) NOT NULL,
    [NoteText] nvarchar(MAX) NOT NULL,
    [CreatedAt] datetime2 NOT NULL DEFAULT (sysutcdatetime()),
    [CreatedByUserID] int NOT NULL,
    [CreatedByName] nvarchar(200) NOT NULL,
    [IsDeleted] bit NOT NULL DEFAULT ((0)),
    [PatientAdmissionID] int NULL,
    [ExternalPatientID] nvarchar(128) NULL,
    [ExternalPatientName] nvarchar(300) NULL
);
GO

IF OBJECT_ID('dbo.APP_DrawerNoteLabelLink', 'U') IS NULL
CREATE TABLE [dbo].[APP_DrawerNoteLabelLink] (
    [NoteID] int NOT NULL,
    [LabelID] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_ExternalApiSettings', 'U') IS NULL
CREATE TABLE [dbo].[APP_ExternalApiSettings] (
    [IntegrationName] nvarchar(50) NOT NULL,
    [BaseUrl] nvarchar(500) NULL,
    [ApiKeyEncrypted] nvarchar(MAX) NULL,
    [TimeoutSeconds] int NOT NULL DEFAULT ((10)),
    [VerifyTls] bit NOT NULL DEFAULT ((1)),
    [Enabled] bit NOT NULL DEFAULT ((0)),
    [LastTestStatus] nvarchar(20) NULL,
    [LastTestMessage] nvarchar(1000) NULL,
    [LastTestAt] datetime NULL,
    [UpdatedAt] datetime NOT NULL DEFAULT (getdate()),
    [UpdatedByUserID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_HardwareConfig', 'U') IS NULL
CREATE TABLE [dbo].[APP_HardwareConfig] (
    [ConfigID] int IDENTITY(1,1) NOT NULL,
    [ConfigKey] nvarchar(100) NOT NULL,
    [ConfigValue] nvarchar(500) NULL,
    [ConfigType] nvarchar(50) NOT NULL DEFAULT ('string'),
    [ConfigGroup] nvarchar(50) NOT NULL,
    [DisplayName] nvarchar(200) NOT NULL,
    [DisplayNameAr] nvarchar(200) NULL,
    [Description] nvarchar(500) NULL,
    [IsEncrypted] bit NOT NULL DEFAULT ((0)),
    [IsEditable] bit NOT NULL DEFAULT ((1)),
    [DisplayOrder] int NOT NULL DEFAULT ((0)),
    [UpdatedAt] datetime2 NULL DEFAULT (getdate()),
    [UpdatedByUserID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_Incident', 'U') IS NULL
CREATE TABLE [dbo].[APP_Incident] (
    [incident_id] int IDENTITY(1,1) NOT NULL,
    [incident_number] varchar(10) NOT NULL,
    [patient_name] nvarchar(255) NULL,
    [primary_doctor_name] nvarchar(255) NULL,
    [feedback_intent_type_id] int NULL,
    [issuing_org_unit_id] int NULL,
    [complaint_summary] nvarchar(2000) NULL,
    [building_id] int NULL,
    [is_inpatient] bit NULL,
    [created_at] datetime2 NOT NULL DEFAULT (sysutcdatetime()),
    [created_by_user_id] int NULL,
    [updated_at] datetime2 NULL,
    [primary_worker_name] nvarchar(255) NULL,
    [updated_by_user_id] int NULL
);
GO

IF OBJECT_ID('dbo.APP_IncidentCase', 'U') IS NULL
CREATE TABLE [dbo].[APP_IncidentCase] (
    [IncidentRequestCaseID] int IDENTITY(1,1) NOT NULL,
    [ComplaintText] nvarchar(MAX) NOT NULL,
    [ImmediateAction] nvarchar(MAX) NOT NULL,
    [TakenAction] nvarchar(MAX) NOT NULL,
    [FeedbackRecievedDate] date NOT NULL,
    [PatientName] nvarchar(200) NOT NULL,
    [IssuingOrgUnitID] int NOT NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL,
    [isINPatient] bit NOT NULL,
    [ClinicalRiskTypeID] int NULL,
    [FeedbackIntentTypeID] int NULL,
    [BuildingID] int NOT NULL,
    [DomainID] int NULL,
    [CategoryID] int NULL,
    [SubCategoryID] int NULL,
    [ClassificationID] int NULL,
    [SeverityID] int NULL,
    [StageID] int NULL,
    [HarmLevelID] int NULL,
    [CaseStatusID] int NOT NULL,
    [SourceID] int NOT NULL,
    [ExplanationStatusID] int NOT NULL,
    [RequiresExplanation] bit NOT NULL DEFAULT ((0)),
    [ForceClosedAt] datetime NULL,
    [ForceClosedByUserID] int NULL,
    [ForceCloseReason] nvarchar(MAX) NULL,
    [incident_id] int NULL,
    [RecordTypeID] int NOT NULL DEFAULT ((1)),
    [IsMorbidity] bit NOT NULL DEFAULT ((0)),
    [UpdatedAt] datetime NULL,
    [IncidentDate] date NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_IncidentCaseDoctor', 'U') IS NULL
CREATE TABLE [dbo].[APP_IncidentCaseDoctor] (
    [IncidentCaseDoctorID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [DoctorID] int NOT NULL,
    [IsPrimary] bit NOT NULL DEFAULT ((0)),
    [AssignedAt] datetime NOT NULL DEFAULT (getdate()),
    [AssignedByUserID] int NOT NULL,
    [DoctorName] nvarchar(200) NULL
);
GO

IF OBJECT_ID('dbo.APP_IncidentCaseEmployee', 'U') IS NULL
CREATE TABLE [dbo].[APP_IncidentCaseEmployee] (
    [EmployeeID] int NOT NULL,
    [FullName] nvarchar(250) NOT NULL,
    [JobTitle] nvarchar(200) NULL,
    [JobID] int NULL,
    [DepartmentID] int NULL,
    [SectionID] int NULL,
    [AdministrationID] int NULL,
    [IsManager] bit NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [IncidentRequestCaseID] int NULL,
    [IsPrimary] bit NULL DEFAULT ((0)),
    [AssignedAt] datetime NULL DEFAULT (getdate()),
    [AssignedByUserID] int NULL,
    [ID] int IDENTITY(1,1) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_IncidentCaseFeedback', 'U') IS NULL
CREATE TABLE [dbo].[APP_IncidentCaseFeedback] (
    [IncidentRequestCaseID] int NOT NULL,
    [Cause_Staff_Training] bit NULL,
    [Cause_Staff_Incentives] bit NULL,
    [Cause_Staff_Competency] bit NULL,
    [Cause_Staff_Understaffed] bit NULL,
    [Cause_Staff_NonCompliance] bit NULL,
    [Cause_Staff_NoCoordination] bit NULL,
    [Cause_Staff_Other] bit NULL,
    [Cause_Staff_OtherText] nvarchar(MAX) NULL,
    [Cause_Process_NotComprehensive] bit NULL,
    [Cause_Process_Unclear] bit NULL,
    [Cause_Process_MissingProtocol] bit NULL,
    [Cause_Process_Other] bit NULL,
    [Cause_Process_OtherText] nvarchar(MAX) NULL,
    [Cause_Equipment_NotAvailable] bit NULL,
    [Cause_Equipment_SystemIncomplete] bit NULL,
    [Cause_Equipment_HardToApply] bit NULL,
    [Cause_Equipment_Other] bit NULL,
    [Cause_Equipment_OtherText] nvarchar(MAX) NULL,
    [Cause_Environment_PlaceNature] bit NULL,
    [Cause_Environment_Surroundings] bit NULL,
    [Cause_Environment_WorkConditions] bit NULL,
    [Cause_Environment_Other] bit NULL,
    [Cause_Environment_OtherText] nvarchar(MAX) NULL,
    [Preventive_MonthlyMeetings] bit NULL,
    [Preventive_TrainingPrograms] bit NULL,
    [Preventive_IncreaseStaff] bit NULL,
    [Preventive_MMCommitteeActions] bit NULL,
    [Preventive_Other] bit NULL,
    [Preventive_OtherText] nvarchar(MAX) NULL,
    [DepartmentExplanationText] nvarchar(MAX) NULL,
    [DepartmentExplanationStatusID] int NOT NULL,
    [DepartmentExplanationReceivalDate] date NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL,
    [AdministrativeSubcaseID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_IncidentCaseSatisfaction', 'U') IS NULL
CREATE TABLE [dbo].[APP_IncidentCaseSatisfaction] (
    [SatisfactionID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [FeedbackNeeded] bit NOT NULL DEFAULT ((0)),
    [FeedbackGiven] bit NOT NULL DEFAULT ((0)),
    [FeedbackDateTime] datetime NULL,
    [SatisfactionStatusID] int NOT NULL,
    [CreatedByUserID] int NOT NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [FeedbackText] nvarchar(1000) NULL
);
GO

IF OBJECT_ID('dbo.APP_IncidentCaseTargetDepartment', 'U') IS NULL
CREATE TABLE [dbo].[APP_IncidentCaseTargetDepartment] (
    [TargetID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [DepartmentID] int NOT NULL,
    [IsPrimary] bit NOT NULL DEFAULT ((0)),
    [AssignedAt] datetime NOT NULL DEFAULT (getdate()),
    [AssignedByUserID] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_BUILDING', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_BUILDING] (
    [BuildingID] int IDENTITY(1,1) NOT NULL,
    [BuildingCode] nvarchar(20) NOT NULL,
    [BuildingName] nvarchar(100) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_CASE_STAGE', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_CASE_STAGE] (
    [StageID] int IDENTITY(1,1) NOT NULL,
    [StageName] nvarchar(100) NOT NULL,
    [StageOrder] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_CASE_STATUS', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_CASE_STATUS] (
    [CaseStatusID] int IDENTITY(1,1) NOT NULL,
    [Code] nvarchar(50) NOT NULL,
    [Name] nvarchar(200) NOT NULL,
    [IsFinal] bit NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [DisplayOrder] int NOT NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_CATEGORY', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_CATEGORY] (
    [CategoryID] int IDENTITY(1,1) NOT NULL,
    [DomainID] int NOT NULL,
    [CategoryName] nvarchar(100) NOT NULL,
    [CategoryOrder] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_CLASSIFICATION', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_CLASSIFICATION] (
    [ClassificationID] int IDENTITY(1,1) NOT NULL,
    [SubCategoryID] int NOT NULL,
    [Classification_AR] nvarchar(300) NOT NULL,
    [Classification_EN] nvarchar(300) NULL,
    [IsActive] bit NOT NULL DEFAULT ((1))
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_CLINICAL_RISK_TYPE', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_CLINICAL_RISK_TYPE] (
    [ClinicalRiskTypeID] int IDENTITY(1,1) NOT NULL,
    [Code] nvarchar(50) NOT NULL,
    [Name] nvarchar(200) NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [DisplayOrder] int NOT NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_DOCTOR', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_DOCTOR] (
    [DoctorID] int NOT NULL,
    [DoctorName] nvarchar(200) NOT NULL,
    [Specialty] nvarchar(200) NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [SourceSystem] nvarchar(100) NULL,
    [LastSyncedAt] datetime NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_DOMAIN', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_DOMAIN] (
    [DomainID] int IDENTITY(1,1) NOT NULL,
    [DomainCode] nvarchar(50) NOT NULL,
    [DomainName] nvarchar(100) NOT NULL,
    [DomainOrder] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_EXPLANATION_STATUS', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_EXPLANATION_STATUS] (
    [StatusID] int IDENTITY(1,1) NOT NULL,
    [StatusName] nvarchar(50) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_FEEDBACK_INTENT_TYPE', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_FEEDBACK_INTENT_TYPE] (
    [FeedbackIntentTypeID] int IDENTITY(1,1) NOT NULL,
    [Code] nvarchar(50) NOT NULL,
    [NameAr] nvarchar(200) NOT NULL,
    [NameEn] nvarchar(200) NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [DisplayOrder] int NOT NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_HARM_LEVEL', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_HARM_LEVEL] (
    [HarmID] int IDENTITY(1,1) NOT NULL,
    [HarmLevel] nvarchar(50) NOT NULL,
    [SeverityOrder] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_RECORD_TYPE', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_RECORD_TYPE] (
    [RecordTypeID] int NOT NULL,
    [TypeName] nvarchar(50) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_Lookup_SatisfactionStatus', 'U') IS NULL
CREATE TABLE [dbo].[APP_Lookup_SatisfactionStatus] (
    [SatisfactionStatusID] int NOT NULL,
    [StatusNameEn] nvarchar(50) NOT NULL,
    [StatusNameAr] nvarchar(50) NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_SEVERITY', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_SEVERITY] (
    [SeverityID] int IDENTITY(1,1) NOT NULL,
    [SeverityCode] nvarchar(20) NOT NULL,
    [SeverityName] nvarchar(50) NOT NULL,
    [SeverityOrder] int NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedBy] int NULL,
    [UpdatedAt] datetime NULL,
    [UpdatedBy] int NULL
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_SOURCE', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_SOURCE] (
    [SourceID] int IDENTITY(1,1) NOT NULL,
    [SourceName] nvarchar(100) NOT NULL,
    [SourceNameAr] nvarchar(100) NOT NULL,
    [DisplayOrder] int NOT NULL,
    [IsActive] bit NULL DEFAULT ((1)),
    [CreatedAt] datetime NULL DEFAULT (getdate()),
    [UpdatedAt] datetime NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_Lookup_SubcaseActionItemStatus', 'U') IS NULL
CREATE TABLE [dbo].[APP_Lookup_SubcaseActionItemStatus] (
    [StatusCode] nvarchar(50) NOT NULL,
    [StatusNameEn] nvarchar(100) NOT NULL,
    [StatusNameAr] nvarchar(100) NULL,
    [DisplayOrder] int NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [IsFinal] bit NOT NULL DEFAULT ((0))
);
GO

IF OBJECT_ID('dbo.APP_Lookup_SubcaseStatus', 'U') IS NULL
CREATE TABLE [dbo].[APP_Lookup_SubcaseStatus] (
    [StatusCode] nvarchar(50) NOT NULL,
    [StatusNameEn] nvarchar(100) NOT NULL,
    [StatusNameAr] nvarchar(100) NULL,
    [DisplayOrder] int NOT NULL,
    [IsFinal] bit NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1))
);
GO

IF OBJECT_ID('dbo.APP_Lookup_SubcaseType', 'U') IS NULL
CREATE TABLE [dbo].[APP_Lookup_SubcaseType] (
    [CaseTypeCode] nvarchar(50) NOT NULL,
    [CaseTypeNameEn] nvarchar(100) NOT NULL,
    [CaseTypeNameAr] nvarchar(100) NULL,
    [IsActive] bit NOT NULL DEFAULT ((1))
);
GO

IF OBJECT_ID('dbo.APP_LOOKUP_SUBCATEGORY', 'U') IS NULL
CREATE TABLE [dbo].[APP_LOOKUP_SUBCATEGORY] (
    [SubCategoryID] int IDENTITY(1,1) NOT NULL,
    [CategoryID] int NOT NULL,
    [SubCategoryName] nvarchar(150) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_OrgUnitPolicy', 'U') IS NULL
CREATE TABLE [dbo].[APP_OrgUnitPolicy] (
    [OrgUnitPolicyID] int IDENTITY(1,1) NOT NULL,
    [OrgUnitID] int NOT NULL,
    [OrgUnitType] int NOT NULL,
    [LowSeverityLimit] int NOT NULL,
    [MediumSeverityLimit] int NOT NULL,
    [HighSeverityLimit] int NOT NULL,
    [ClinicalDomainLimit] int NOT NULL,
    [ManagementDomainLimit] int NOT NULL,
    [RelationalDomainLimit] int NOT NULL,
    [EnableLowSeverityRepetitionRule] bit NOT NULL DEFAULT ((0)),
    [EnableMediumSeverityRepetitionRule] bit NOT NULL DEFAULT ((0)),
    [EnableHighSeverityPercentageRule] bit NOT NULL DEFAULT ((0)),
    [EnableHighSeverityPercentageByDomainRule] bit NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL,
    [UpdatedAt] datetime NULL,
    [UpdatedByUserID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_PublicationBatch', 'U') IS NULL
CREATE TABLE [dbo].[APP_PublicationBatch] (
    [PublicationBatchID] int IDENTITY(1,1) NOT NULL,
    [PublicationSerial] int NOT NULL,
    [PublishedAt] datetime NOT NULL DEFAULT (getdate()),
    [PublishedByUserID] int NOT NULL,
    [CasesPublishedCount] int NOT NULL,
    [Notes] nvarchar(MAX) NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_PublicationBatchCase', 'U') IS NULL
CREATE TABLE [dbo].[APP_PublicationBatchCase] (
    [PublicationBatchCaseID] int IDENTITY(1,1) NOT NULL,
    [PublicationBatchID] int NOT NULL,
    [IncidentCaseID] int NOT NULL,
    [AdministrativeSubcaseID] int NULL,
    [TargetOrgUnitID] int NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_RCAFactorCategory', 'U') IS NULL
CREATE TABLE [dbo].[APP_RCAFactorCategory] (
    [CategoryID] int IDENTITY(1,1) NOT NULL,
    [CategoryCode] nvarchar(100) NOT NULL,
    [CategoryNameEn] nvarchar(255) NOT NULL,
    [CategoryNameAr] nvarchar(255) NULL,
    [SortOrder] int NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime2 NOT NULL DEFAULT (sysutcdatetime()),
    [CreatedByUserID] int NULL,
    [UpdatedAt] datetime2 NULL,
    [UpdatedByUserID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_RCASuggestion', 'U') IS NULL
CREATE TABLE [dbo].[APP_RCASuggestion] (
    [SuggestionID] int IDENTITY(1,1) NOT NULL,
    [CategoryID] int NOT NULL,
    [SuggestionType] nvarchar(50) NOT NULL,
    [SuggestionTextEn] nvarchar(500) NULL,
    [SuggestionTextAr] nvarchar(500) NOT NULL,
    [DescriptionEn] nvarchar(MAX) NULL,
    [DescriptionAr] nvarchar(MAX) NULL,
    [SortOrder] int NOT NULL DEFAULT ((0)),
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime2 NOT NULL DEFAULT (sysutcdatetime()),
    [CreatedByUserID] int NULL,
    [UpdatedAt] datetime2 NULL,
    [UpdatedByUserID] int NULL,
    [PairedSuggestionID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_ReportConfig', 'U') IS NULL
CREATE TABLE [dbo].[APP_ReportConfig] (
    [ConfigKey] nvarchar(100) NOT NULL,
    [ConfigValue] nvarchar(MAX) NOT NULL DEFAULT (''),
    [UpdatedAt] datetime NOT NULL DEFAULT (getdate()),
    [UpdatedBy] int NULL
);
GO

IF OBJECT_ID('dbo.APP_RESERVE_DOCTOR', 'U') IS NULL
CREATE TABLE [dbo].[APP_RESERVE_DOCTOR] (
    [DoctorID] int IDENTITY(1,1) NOT NULL,
    [DoctorName] nvarchar(200) NOT NULL,
    [Specialty] nvarchar(200) NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [SourceSystem] nvarchar(100) NULL DEFAULT ('MANUAL'),
    [LastSyncedAt] datetime NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_RESERVE_PATIENT', 'U') IS NULL
CREATE TABLE [dbo].[APP_RESERVE_PATIENT] (
    [PatientAdmissionID] int IDENTITY(1,1) NOT NULL,
    [CaseID] int NULL,
    [AdmissionDate] datetime NULL,
    [AdmissionTime] datetime NULL,
    [DischargeDate] datetime NULL,
    [FullName] nvarchar(250) NULL,
    [FirstName] nvarchar(150) NULL,
    [MiddleName] nvarchar(150) NULL,
    [LastName] nvarchar(150) NULL,
    [Spouse] nvarchar(150) NULL,
    [MotherName] nvarchar(150) NULL,
    [AddressLine1] nvarchar(300) NULL,
    [AddressLine2] nvarchar(300) NULL,
    [PhoneNumber1] nvarchar(50) NULL,
    [PhoneNumber2] nvarchar(50) NULL,
    [SEX] nvarchar(10) NULL,
    [BirthDate] date NULL,
    [DocumentNumber] nvarchar(100) NULL,
    [DischargeNumber] nvarchar(100) NULL,
    [Reference] nvarchar(200) NULL,
    [Diagnoses] nvarchar(MAX) NULL,
    [DoctorID] int NULL,
    [AssistantDoctorID] int NULL,
    [MedicalServiceID] int NULL,
    [MedicalServiceClassID] int NULL,
    [RoomNumber] nvarchar(50) NULL,
    [BED] nvarchar(50) NULL,
    [Stay] int NULL,
    [SerialNumber] nvarchar(100) NULL,
    [SerialSequence] nvarchar(100) NULL,
    [MedicalFileNumber] nvarchar(100) NULL,
    [MaritalStatusID] int NULL,
    [EmployeeID] int NULL,
    [NationalityID] int NULL,
    [LocationTypeID] int NULL,
    [RoomBedID] int NULL,
    [CareCenterID] int NULL,
    [GuarantorCode] nvarchar(100) NULL,
    [GuarantorDocumentNumber] nvarchar(100) NULL,
    [GuarantorClassNumber] nvarchar(100) NULL,
    [SecondGuarantorCode] nvarchar(100) NULL,
    [SecondGuarantorDocumentNumber] nvarchar(100) NULL,
    [SecondGuarantorClassNumber] nvarchar(100) NULL,
    [ThirdGuarantorCode] nvarchar(100) NULL,
    [ThirdGuarantorDocumentNumber] nvarchar(100) NULL,
    [ThirdGuarantorClassNumber] nvarchar(100) NULL,
    [GuarantorAmount] decimal(18,2) NULL,
    [SecondGuarantorAmount] decimal(18,2) NULL,
    [ThirdGuarantorAmount] decimal(18,2) NULL,
    [GuarantorGroupID] int NULL,
    [GuarantorGroupName] nvarchar(150) NULL,
    [GuarantorSkippedAmount] decimal(18,2) NULL,
    [GovernmentRecordNumber] nvarchar(100) NULL,
    [Section] nvarchar(150) NULL,
    [Building] nvarchar(150) NULL,
    [FILE] nvarchar(255) NULL,
    [T-Moh] nvarchar(100) NULL,
    [CostSample] decimal(18,2) NULL,
    [BedTypeID] int NULL,
    [SystemTime] datetime NULL DEFAULT (getdate()),
    [IsActive] bit NOT NULL DEFAULT ((1))
);
GO

IF OBJECT_ID('dbo.APP_Roles', 'U') IS NULL
CREATE TABLE [dbo].[APP_Roles] (
    [RoleID] int IDENTITY(1,1) NOT NULL,
    [RoleCode] nvarchar(50) NOT NULL,
    [RoleNameEn] nvarchar(100) NOT NULL,
    [RoleNameAr] nvarchar(100) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_SeasonalOrgUnitReport', 'U') IS NULL
CREATE TABLE [dbo].[APP_SeasonalOrgUnitReport] (
    [SeasonalReportID] int IDENTITY(1,1) NOT NULL,
    [SeasonID] int NOT NULL,
    [OrgUnitID] int NOT NULL,
    [OrgUnitType] int NOT NULL,
    [TotalCases] int NOT NULL DEFAULT ((0)),
    [LowSeverityCount] int NOT NULL DEFAULT ((0)),
    [MediumSeverityCount] int NOT NULL DEFAULT ((0)),
    [HighSeverityCount] int NOT NULL DEFAULT ((0)),
    [ClinicalDomainCount] int NOT NULL DEFAULT ((0)),
    [ManagementDomainCount] int NOT NULL DEFAULT ((0)),
    [RelationalDomainCount] int NOT NULL DEFAULT ((0)),
    [IsCompliant] bit NOT NULL DEFAULT ((1)),
    [ViolatedRules] nvarchar(MAX) NULL,
    [EvaluatedAt] datetime NOT NULL DEFAULT (getdate()),
    [ExplanationStatusID] int NOT NULL,
    [ExplanationText] nvarchar(MAX) NULL,
    [ExplanationSubmittedAt] datetime NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_SeasonalOrgUnitReport_ClassificationStats', 'U') IS NULL
CREATE TABLE [dbo].[APP_SeasonalOrgUnitReport_ClassificationStats] (
    [StatID] int IDENTITY(1,1) NOT NULL,
    [SeasonalReportID] int NOT NULL,
    [DomainID] int NOT NULL,
    [CategoryID] int NOT NULL,
    [SubCategoryID] int NOT NULL,
    [ClassificationID] int NOT NULL,
    [TotalCount] int NOT NULL,
    [LowCount] int NOT NULL,
    [MediumCount] int NOT NULL,
    [HighCount] int NOT NULL,
    [PreventiveYesCount] int NOT NULL,
    [PreventiveNoCount] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_SeasonalOrgUnitReport_PolicySnapshot', 'U') IS NULL
CREATE TABLE [dbo].[APP_SeasonalOrgUnitReport_PolicySnapshot] (
    [PolicySnapshotID] int IDENTITY(1,1) NOT NULL,
    [SeasonalReportID] int NOT NULL,
    [LowSeverityLimit] int NOT NULL,
    [MediumSeverityLimit] int NOT NULL,
    [HighSeverityLimit] int NOT NULL,
    [ClinicalDomainLimit] int NOT NULL,
    [ManagementDomainLimit] int NOT NULL,
    [RelationalDomainLimit] int NOT NULL,
    [EnableLowSeverityRepetitionRule] bit NOT NULL,
    [EnableMediumSeverityRepetitionRule] bit NOT NULL,
    [EnableHighSeverityPercentageRule] bit NOT NULL,
    [EnableHighSeverityPercentageByDomainRule] bit NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_SeasonCase', 'U') IS NULL
CREATE TABLE [dbo].[APP_SeasonCase] (
    [SeasonCaseID] int IDENTITY(1,1) NOT NULL,
    [SeasonID] int NOT NULL,
    [DepartmentID] int NOT NULL,
    [SeasonalReportText] nvarchar(MAX) NULL,
    [SeasonalReportDepartmentFeedback] nvarchar(MAX) NULL,
    [SeasonCaseStatusID] int NOT NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_SubcaseActionItem', 'U') IS NULL
CREATE TABLE [dbo].[APP_SubcaseActionItem] (
    [ActionItemID] int IDENTITY(1,1) NOT NULL,
    [SubcaseID] int NOT NULL,
    [Status] nvarchar(50) NOT NULL,
    [Title] nvarchar(300) NOT NULL,
    [Description] nvarchar(MAX) NULL,
    [DueDate] date NULL,
    [AssignedToUserID] int NULL,
    [StartedAt] datetime NULL,
    [CompletedAt] datetime NULL,
    [VerifiedAt] datetime NULL,
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CreatedByUserID] int NOT NULL,
    [UpdatedAt] datetime NULL,
    [UpdatedByUserID] int NULL,
    [EnteredByUserID] int NULL,
    [EnteredForRole] nvarchar(50) NULL,
    [EntryMode] nvarchar(50) NULL,
    [EntryTimestamp] datetime NULL
);
GO

IF OBJECT_ID('dbo.APP_SubcaseActionItemChangeNotice', 'U') IS NULL
CREATE TABLE [dbo].[APP_SubcaseActionItemChangeNotice] (
    [NoticeID] int IDENTITY(1,1) NOT NULL,
    [ActionItemID] int NOT NULL,
    [RecipientUserID] int NOT NULL,
    [OldTitle] nvarchar(300) NOT NULL,
    [NewTitle] nvarchar(300) NOT NULL,
    [OldDescription] nvarchar(MAX) NULL,
    [NewDescription] nvarchar(MAX) NULL,
    [OldDueDate] date NULL,
    [NewDueDate] date NULL,
    [ChangedByUserID] int NOT NULL,
    [ChangedAt] datetime NOT NULL DEFAULT (getdate()),
    [AcknowledgedAt] datetime NULL,
    [AcknowledgedByUserID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_SubcaseDecisionAcknowledgment', 'U') IS NULL
CREATE TABLE [dbo].[APP_SubcaseDecisionAcknowledgment] (
    [AcknowledgmentID] int IDENTITY(1,1) NOT NULL,
    [SubcaseID] int NOT NULL,
    [OrgLevel] varchar(20) NOT NULL,
    [AcknowledgedByUserID] int NOT NULL,
    [AcknowledgedAt] datetime NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('dbo.APP_SubcaseRCASuggestionSelection', 'U') IS NULL
CREATE TABLE [dbo].[APP_SubcaseRCASuggestionSelection] (
    [SelectionID] int IDENTITY(1,1) NOT NULL,
    [SubcaseID] int NOT NULL,
    [SuggestionID] int NOT NULL,
    [SelectedByUserID] int NULL,
    [SelectedAt] datetime2 NOT NULL DEFAULT (sysutcdatetime())
);
GO

IF OBJECT_ID('dbo.APP_SupervisorActionItem', 'U') IS NULL
CREATE TABLE [dbo].[APP_SupervisorActionItem] (
    [ActionItemID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [SubcaseID] int NULL,
    [TargetOrgUnitID] int NOT NULL,
    [TargetUserID] int NULL,
    [CreatedByUserID] int NOT NULL,
    [CreatedByRoleCode] nvarchar(50) NOT NULL,
    [Description] nvarchar(MAX) NOT NULL,
    [DueDate] date NULL,
    [Status] nvarchar(20) NOT NULL DEFAULT ('PENDING'),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [CompletedAt] datetime NULL,
    [CancelledAt] datetime NULL,
    [UpdatedAt] datetime NULL,
    [UpdatedByUserID] int NULL,
    [AcknowledgedAt] datetime NULL,
    [AcknowledgedByUserID] int NULL
);
GO

IF OBJECT_ID('dbo.APP_SupervisorActionItemAuditLog', 'U') IS NULL
CREATE TABLE [dbo].[APP_SupervisorActionItemAuditLog] (
    [AuditLogID] int IDENTITY(1,1) NOT NULL,
    [ActionItemID] int NOT NULL,
    [Action] nvarchar(20) NOT NULL,
    [PerformedByUserID] int NOT NULL,
    [PerformedAt] datetime NOT NULL DEFAULT (getdate()),
    [Note] nvarchar(500) NULL
);
GO

IF OBJECT_ID('dbo.APP_SystemSettings', 'U') IS NULL
CREATE TABLE [dbo].[APP_SystemSettings] (
    [SettingID] int IDENTITY(1,1) NOT NULL,
    [SettingKey] nvarchar(100) NOT NULL,
    [SettingValue] nvarchar(MAX) NULL,
    [SettingLabel] nvarchar(200) NULL,
    [SettingLabelAr] nvarchar(200) NULL,
    [SettingType] nvarchar(50) NULL,
    [Description] nvarchar(500) NULL,
    [DescriptionAr] nvarchar(500) NULL,
    [IsActive] bit NULL DEFAULT ((1)),
    [CreatedAt] datetime NULL DEFAULT (getdate()),
    [UpdatedAt] datetime NULL DEFAULT (getdate()),
    [UpdatedBy] int NULL
);
GO

IF OBJECT_ID('dbo.APP_UserRoleScope', 'U') IS NULL
CREATE TABLE [dbo].[APP_UserRoleScope] (
    [UserRoleScopeID] int IDENTITY(1,1) NOT NULL,
    [UserID] int NOT NULL,
    [RoleID] int NOT NULL,
    [OrgUnitID] int NOT NULL,
    [OrgUnitType] nvarchar(50) NOT NULL
);
GO

IF OBJECT_ID('dbo.APP_Users', 'U') IS NULL
CREATE TABLE [dbo].[APP_Users] (
    [UserID] int IDENTITY(1,1) NOT NULL,
    [Username] nvarchar(100) NOT NULL,
    [PasswordHash] nvarchar(255) NOT NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [CreatedAt] datetime NOT NULL DEFAULT (getdate()),
    [DisplayName] nvarchar(150) NULL,
    [DepartmentDisplayName] nvarchar(150) NULL,
    [Email] nvarchar(255) NULL
);
GO

IF OBJECT_ID('dbo.IncidentRequest', 'U') IS NULL
CREATE TABLE [dbo].[IncidentRequest] (
    [UniqueID] int IDENTITY(1,1) NOT NULL,
    [YearCounter] int NULL,
    [IncidentRequesterTypeID] int NULL,
    [Code] nvarchar(50) NULL,
    [PatientTypeID] int NULL,
    [DoctorID] nvarchar(8) NULL,
    [EmployeeID] int NULL,
    [MRN] bigint NULL,
    [SourceBuilding] nvarchar(3) NULL,
    [PatientName] nvarchar(200) NULL,
    [PatientID] bigint NULL,
    [DateAndTimeRecieved] datetime NOT NULL,
    [IncidentStatusID] int NULL,
    [IncidentSourceID] int NULL,
    [SourceSectionID] int NULL,
    [SourceDepartmentID] int NULL,
    [SourceDepartmentName] nvarchar(100) NULL,
    [SourceAdminID] int NULL,
    [SourceAdminName] nvarchar(100) NULL,
    [RequesterName] nvarchar(150) NULL,
    [Note] nvarchar(1000) NULL,
    [IsFeedbackRequested] bit NULL,
    [IsFeedbackGiven] bit NULL,
    [IsInPatient] bit NULL,
    [DateAndTimeFeedbackGiven] datetime NULL,
    [SatisfactoryID] int NULL,
    [DateAndTimeCreated] datetime NULL,
    [DateAndTimeUpdated] datetime NULL,
    [CreatedByApplicationUserID] int NULL,
    [UpdatedByApplicationUserID] int NULL,
    [Frozen] bit NULL
);
GO

IF OBJECT_ID('dbo.IncidentRequestCase', 'U') IS NULL
CREATE TABLE [dbo].[IncidentRequestCase] (
    [UniqueID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestID] int NOT NULL,
    [Description] nvarchar(1000) NULL,
    [IncidentCaseCategoryID] int NULL,
    [IncidentCaseSubCategoryID] int NULL,
    [IncidentRequestCaseStatusID] int NULL,
    [CaseBuilding] nvarchar(3) NULL,
    [Note] nvarchar(MAX) NULL,
    [DoctorID] int NULL,
    [DoctorSpecility] nvarchar(200) NULL,
    [SectionID] int NULL,
    [DepartmentID] int NULL,
    [DeprtmentName] nvarchar(100) NULL,
    [AdminID] int NULL,
    [AdminName] nvarchar(100) NULL,
    [DateAndTimeCreated] datetime NULL,
    [DateAndTimeUpdated] datetime NULL,
    [CreatedByApplicationUserID] int NULL,
    [UpdatedByApplicationUserID] int NULL,
    [DateAndTimeHappened] datetime NULL,
    [IncidentTypeID] int NULL,
    [Frozen] bit NULL,
    [oldsectionID] int NULL
);
GO

IF OBJECT_ID('dbo.IncidentRequestCaseAction', 'U') IS NULL
CREATE TABLE [dbo].[IncidentRequestCaseAction] (
    [UniqueID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [Description] nvarchar(MAX) NULL,
    [IncidentRequestCaseActionMessageID] int NULL,
    [IncidentRequestCaseActionInstructionID] int NULL,
    [Note] nvarchar(MAX) NULL,
    [SectionNote] nvarchar(MAX) NULL,
    [DateAndTimeCreated] datetime NULL,
    [DateAndTimeUpdated] datetime NULL,
    [CreatedByApplicationUserID] int NULL,
    [UpdatedByApplicationUserID] int NULL,
    [IsAutoUpdated] bit NULL,
    [SelectionNoteID] int NULL,
    [SelectionNote] nvarchar(MAX) NULL,
    [JobDescriptionID] int NULL,
    [JobDescriptionDetailID] int NULL,
    [DateAndTimeSelectionNoteSaved] datetime NULL,
    [SavedSelectionNoteByEmployeeID] int NULL,
    [IsAutoSelectionNoteSaved] bit NULL,
    [DateAndTimeSectionNoteSaved] datetime NULL,
    [SavedSectionNoteByEmployeeID] int NULL,
    [IsAutoSectionNoteSaved] bit NULL,
    [DepartmentNote] nvarchar(MAX) NULL,
    [SavedDepartmentNoteByEmployeeID] int NULL,
    [DateAndTimeDepartmentNoteSaved] datetime NULL,
    [IsAutoDepartmentNoteSaved] bit NULL,
    [SubmitByEmployeeID] int NULL,
    [DateAndTimeSubmit] datetime NULL,
    [Locked] bit NULL,
    [ProblemReason] nvarchar(MAX) NULL,
    [ReasonNotResolvingID] int NULL,
    [GoverningPolicies] nvarchar(MAX) NULL,
    [IsImprovementForm] bit NULL,
    [Frozen] bit NULL
);
GO

IF OBJECT_ID('dbo.Instance', 'U') IS NULL
CREATE TABLE [dbo].[Instance] (
    [UniqueID] int NOT NULL,
    [EnglishName] nvarchar(100) NOT NULL,
    [ParialEnglishName] nvarchar(100) NULL,
    [ArabicName] nvarchar(100) NOT NULL,
    [ParialArabicName] nvarchar(100) NULL,
    [DownloadLocation] nvarchar(1000) NOT NULL,
    [PublishLocation] nvarchar(1000) NOT NULL,
    [DamanContractNumber] int NULL,
    [Address] nvarchar(200) NULL,
    [Logo] image NULL,
    [IsMain] bit NULL,
    [CobolServerLinK] nvarchar(50) NULL,
    [ArchiveFolderPath] nvarchar(100) NULL
);
GO

IF OBJECT_ID('dbo.Parameter', 'U') IS NULL
CREATE TABLE [dbo].[Parameter] (
    [UniqueID] int IDENTITY(1,1) NOT NULL,
    [Description] nvarchar(200) NULL,
    [ParentID] int NULL,
    [Frozen] bit NULL,
    [CreateDate] datetime NULL,
    [CreatedID] int NULL,
    [Severity] int NULL
);
GO

IF OBJECT_ID('dbo.Role', 'U') IS NULL
CREATE TABLE [dbo].[Role] (
    [UniqueID] nchar(10) NOT NULL,
    [Name] nvarchar(50) NULL,
    [IsActive] bit NULL
);
GO

IF OBJECT_ID('dbo.SchemaMigrationHistory', 'U') IS NULL
CREATE TABLE [dbo].[SchemaMigrationHistory] (
    [MigrationID] int IDENTITY(1,1) NOT NULL,
    [MigrationName] nvarchar(255) NOT NULL,
    [Checksum] nvarchar(128) NULL,
    [AppliedAt] datetime2 NOT NULL DEFAULT (getdate()),
    [AppliedBy] nvarchar(200) NULL,
    [ApplicationVersion] nvarchar(50) NULL,
    [Success] bit NOT NULL DEFAULT ((1))
);
GO

IF OBJECT_ID('dbo.Season', 'U') IS NULL
CREATE TABLE [dbo].[Season] (
    [UniqueID] int NOT NULL,
    [SeasonName] nvarchar(100) NULL,
    [StartDate] date NULL,
    [EndDate] date NULL,
    [IsDone] bit NULL,
    [Frozen] bit NULL,
    [CreateDate] datetime NULL,
    [CreateID] int NULL
);
GO

IF OBJECT_ID('dbo.UserRole', 'U') IS NULL
CREATE TABLE [dbo].[UserRole] (
    [UniqueID] int NOT NULL,
    [UserID] int NULL,
    [RoleID] int NULL,
    [Frozen] bit NULL
);
GO

IF OBJECT_ID('dbo.Users', 'U') IS NULL
CREATE TABLE [dbo].[Users] (
    [UniqueID] int IDENTITY(1,1) NOT NULL,
    [Name] nvarchar(150) NULL,
    [LoginName] nvarchar(50) NULL,
    [Password] nvarchar(10) NULL,
    [Frozen] bit NULL,
    [CreatedBYID] int NULL,
    [CreatedDate] datetime NULL,
    [IsAdmin] bit NULL,
    [IsIncidentUser] bit NULL,
    [IsViewer] bit NULL,
    [RoleID] int NULL,
    [UserID] int NULL
);
GO

IF OBJECT_ID('dbo.VW_HrEmployeeProfileView', 'U') IS NULL
CREATE TABLE [dbo].[VW_HrEmployeeProfileView] (
    [EmployeeID] int NOT NULL,
    [FullName] nvarchar(250) NULL,
    [JobTitle] nvarchar(200) NULL,
    [JobID] int NULL,
    [DepartmentID] int NULL,
    [SectionID] int NULL,
    [AdministrationID] int NULL,
    [IsManager] bit NULL,
    [IsActive] bit NULL
);
GO

IF OBJECT_ID('ml.CaseTrainingRecord', 'U') IS NULL
CREATE TABLE [ml].[CaseTrainingRecord] (
    [CaseTrainingRecordID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [ComplaintText] nvarchar(MAX) NULL,
    [ImmediateActionText] nvarchar(MAX) NULL,
    [TakenActionText] nvarchar(MAX) NULL,
    [FeedbackTypeID] int NULL,
    [DomainID] int NULL,
    [CategoryID] int NULL,
    [SubCategoryID] int NULL,
    [ClassificationID] int NULL,
    [SeverityLevelID] int NULL,
    [StageID] int NULL,
    [HarmLevelID] int NULL,
    [ImprovementOpportunityTypeID] int NULL,
    [ComplaintEmbedding] varbinary(MAX) NULL,
    [CombinedTextEmbedding] varbinary(MAX) NULL,
    [EmbeddingModelVersionID] int NULL,
    [EmbeddingDimension] int NULL,
    [ProcessingStatus] nvarchar(20) NOT NULL DEFAULT ('Pending'),
    [LastProcessedAt] datetime2 NULL,
    [SourceDataUpdatedAt] datetime2 NULL,
    [CreatedAt] datetime2 NOT NULL DEFAULT (getdate()),
    [UpdatedAt] datetime2 NULL
);
GO

IF OBJECT_ID('ml.EmbeddingModelVersion', 'U') IS NULL
CREATE TABLE [ml].[EmbeddingModelVersion] (
    [EmbeddingModelVersionID] int IDENTITY(1,1) NOT NULL,
    [ModelName] nvarchar(200) NOT NULL,
    [ModelPathOrIdentifier] nvarchar(500) NOT NULL,
    [ModelArchitecture] nvarchar(200) NULL,
    [ModelChecksum] nvarchar(128) NULL,
    [EmbeddingDimension] int NOT NULL,
    [PoolingMethod] nvarchar(50) NULL,
    [NormalizationMethod] nvarchar(50) NULL,
    [TokenizerIdentifier] nvarchar(200) NULL,
    [ActivatedAt] datetime2 NOT NULL DEFAULT (getdate()),
    [RetiredAt] datetime2 NULL,
    [IsActive] bit NOT NULL DEFAULT ((1)),
    [ConfigurationJson] nvarchar(MAX) NULL
);
GO

IF OBJECT_ID('ml.EmbeddingProcessingJob', 'U') IS NULL
CREATE TABLE [ml].[EmbeddingProcessingJob] (
    [EmbeddingProcessingJobID] int IDENTITY(1,1) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [JobType] nvarchar(30) NOT NULL,
    [Status] nvarchar(20) NOT NULL DEFAULT ('Pending'),
    [AttemptCount] int NOT NULL DEFAULT ((0)),
    [MaximumAttempts] int NOT NULL DEFAULT ((5)),
    [RequestedAt] datetime2 NOT NULL DEFAULT (getdate()),
    [StartedAt] datetime2 NULL,
    [CompletedAt] datetime2 NULL,
    [NextRetryAt] datetime2 NULL,
    [LastErrorCode] nvarchar(50) NULL,
    [LastErrorMessage] nvarchar(MAX) NULL,
    [WorkerID] nvarchar(100) NULL,
    [EmbeddingModelVersionID] int NULL,
    [ImportBatchID] int NULL
);
GO

IF OBJECT_ID('ml.HistoricalTrainingExample', 'U') IS NULL
CREATE TABLE [ml].[HistoricalTrainingExample] (
    [HistoricalTrainingExampleID] int IDENTITY(1,1) NOT NULL,
    [LegacySource] nvarchar(100) NULL,
    [LegacySourceTable] nvarchar(200) NULL,
    [LegacySourceRowID] int NULL,
    [PossibleIncidentRequestCaseID] int NULL,
    [LinkConfidence] nvarchar(20) NULL,
    [ComplaintText] nvarchar(MAX) NULL,
    [ImmediateActionText] nvarchar(MAX) NULL,
    [TakenActionText] nvarchar(MAX) NULL,
    [FeedbackTypeID] int NULL,
    [DomainID] int NULL,
    [CategoryID] int NULL,
    [SubCategoryID] int NULL,
    [ClassificationID] int NULL,
    [SeverityLevelID] int NULL,
    [StageID] int NULL,
    [HarmLevelID] int NULL,
    [ImprovementOpportunityTypeID] int NULL,
    [EmbeddingText1] varbinary(MAX) NULL,
    [EmbeddingText2] varbinary(MAX) NULL,
    [EmbeddingText3] varbinary(MAX) NULL,
    [EmbeddingText123] varbinary(MAX) NULL,
    [EmbeddingText23] varbinary(MAX) NULL,
    [SentenceEmbedding1] varbinary(MAX) NULL,
    [SentenceEmbedding2] varbinary(MAX) NULL,
    [SentenceEmbedding3] varbinary(MAX) NULL,
    [SentenceEmbedding4] varbinary(MAX) NULL,
    [SentenceEmbedding5] varbinary(MAX) NULL,
    [SentenceEmbedding6] varbinary(MAX) NULL,
    [ImportedAt] datetime2 NOT NULL DEFAULT (getdate()),
    [MigrationBatchID] nvarchar(100) NULL,
    [PreservationNotes] nvarchar(MAX) NULL
);
GO

IF OBJECT_ID('ml.ImportBatch', 'U') IS NULL
CREATE TABLE [ml].[ImportBatch] (
    [ImportBatchID] int IDENTITY(1,1) NOT NULL,
    [OriginalFileName] nvarchar(500) NULL,
    [FileChecksum] nvarchar(128) NULL,
    [TemplateVersion] nvarchar(50) NULL,
    [UploadedByUserID] int NULL,
    [UploadedAt] datetime2 NOT NULL DEFAULT (getdate()),
    [Status] nvarchar(20) NOT NULL DEFAULT ('Processing'),
    [TotalRows] int NULL,
    [AcceptedRows] int NULL,
    [RejectedRows] int NULL,
    [DuplicateRows] int NULL,
    [CreatedCaseCount] int NULL,
    [MLCompletedCount] int NULL,
    [MLFailedCount] int NULL,
    [CompletedAt] datetime2 NULL
);
GO

IF OBJECT_ID('ml.ImportSourceRecordMap', 'U') IS NULL
CREATE TABLE [ml].[ImportSourceRecordMap] (
    [ImportSourceRecordMapID] int IDENTITY(1,1) NOT NULL,
    [ImportBatchID] int NULL,
    [ExternalSourceSystem] nvarchar(100) NOT NULL,
    [ExternalRecordID] nvarchar(200) NOT NULL,
    [IncidentRequestCaseID] int NOT NULL,
    [ImportedAt] datetime2 NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('ml.LegacyDbSizeHistory', 'U') IS NULL
CREATE TABLE [ml].[LegacyDbSizeHistory] (
    [LegacyDbSizeHistoryID] int IDENTITY(1,1) NOT NULL,
    [RecordDate] nvarchar(50) NOT NULL,
    [RecordCount] int NOT NULL,
    [LegacyRecordedAt] nvarchar(50) NULL,
    [MigratedAt] datetime2 NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('ml.LegacyModelMetricHistory', 'U') IS NULL
CREATE TABLE [ml].[LegacyModelMetricHistory] (
    [LegacyModelMetricHistoryID] int IDENTITY(1,1) NOT NULL,
    [LegacyMetricID] int NULL,
    [RunID] nvarchar(100) NULL,
    [ModelName] nvarchar(200) NULL,
    [NumRecords] int NULL,
    [Accuracy] float NULL,
    [Precision_] float NULL,
    [Recall_] float NULL,
    [F1] float NULL,
    [LastTrained] nvarchar(50) NULL,
    [MigratedAt] datetime2 NOT NULL DEFAULT (getdate())
);
GO

IF OBJECT_ID('ml.LegacyTrainingRunHistory', 'U') IS NULL
CREATE TABLE [ml].[LegacyTrainingRunHistory] (
    [LegacyTrainingRunHistoryID] int IDENTITY(1,1) NOT NULL,
    [RunID] nvarchar(100) NOT NULL,
    [StartedAt] nvarchar(50) NULL,
    [FinishedAt] nvarchar(50) NULL,
    [Status] nvarchar(50) NULL,
    [ModelsTrained] int NULL,
    [LegacyCreatedAt] nvarchar(50) NULL,
    [MigratedAt] datetime2 NOT NULL DEFAULT (getdate())
);
GO
