============================================================  
TABLE: AdminsrationUnit  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - Name: nvarchar(50)  
  - ParentID: int  
  - Frozen: bit (Default=((0)))  
  - Type: int  
  - CreateDate: datetime  
  - CreateID: int  
  - UpdateDate: datetime  
  - UpdateUser: int  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  - ParentID -> AdminsrationUnit.UniqueID  
  
  
============================================================  
TABLE: AdminsrationUnitHistory  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - AdminID: int  
  - Name: nvarchar(50)  
  - ParentID: int  
  - Frozen: bit  
  - CreateDate: datetime  
  - CreateID: int  
  - UpdateDate: datetime  
  - UpdateUser: int  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_ActionItem  
============================================================  
COLUMNS:  
  - ActionItemID: int (NOT NULL)  
  - IncidentRequestCaseID: int  
  - SeasonCaseID: int  
  - ActionTitle: nvarchar(300) (NOT NULL)  
  - ActionDescription: nvarchar(-1)  
  - DueDate: date  
  - DateSubmitted: date  
  - IsDone: bit (NOT NULL) (Default=((0)))  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  - CreatedByUserID: int (NOT NULL)  
  
PRIMARY KEY:  
  - ActionItemID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_DepartmentEvaluationRule  
============================================================  
COLUMNS:  
  - DepartmentEvaluationRuleID: int (NOT NULL)  
  - DepartmentID: int (NOT NULL)  
  - LowSeverityMaxCount: int  
  - MediumSeverityMaxCount: int  
  - HighSeverityMaxCount: int  
  - HighSeverityMaxPercentage: decimal  
  - HighSeverityByDomainMaxPercentage: decimal  
  - EnableLowSeverityRepetitionRule: bit (NOT NULL) (Default=((0)))  
  - EnableMediumSeverityRepetitionRule: bit (NOT NULL) (Default=((0)))  
  - EnableHighSeverityPercentageRule: bit (NOT NULL) (Default=((0)))  
  - EnableHighSeverityPercentageByDomainRule: bit (NOT NULL) (Default=((0)))  
  - IsActive: bit (NOT NULL) (Default=((1)))  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  - CreatedByUserID: int (NOT NULL)  
  
PRIMARY KEY:  
  - DepartmentEvaluationRuleID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_IncidentCase  
============================================================  
COLUMNS:  
  - IncidentRequestCaseID: int (NOT NULL)  
  - ComplaintText: nvarchar(-1) (NOT NULL)  
  - ImmediateAction: nvarchar(-1)  
  - TakenAction: nvarchar(-1)  
  - FeedbackRecievedDate: date  
  - PatientName: nvarchar(200)  
  - DoctorName: nvarchar(200)  
  - DoctorID: int  
  - IssuingOrgUnitID: int (NOT NULL)  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  - CreatedByUserID: int (NOT NULL)  
  - InOut: nvarchar(50)  
  - ClinicalRiskTypeID: int (NOT NULL)  
  - FeedbackIntentTypeID: int (NOT NULL)  
  - BuildingID: int  
  - DomainID: int (NOT NULL)  
  - CategoryID: int (NOT NULL)  
  - SubCategoryID: int (NOT NULL)  
  - ClassificationID: int (NOT NULL)  
  - SeverityID: int  
  - StageID: int (NOT NULL)  
  - HarmLevelID: int (NOT NULL)  
  - CaseStatusID: int (NOT NULL)  
  
PRIMARY KEY:  
  - IncidentRequestCaseID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_IncidentCaseDoctor  
============================================================  
COLUMNS:  
  - IncidentCaseDoctorID: int (NOT NULL)  
  - IncidentRequestCaseID: int (NOT NULL)  
  - DoctorID: int (NOT NULL)  
  - IsPrimary: bit (NOT NULL) (Default=((0)))  
  - AssignedAt: datetime (NOT NULL) (Default=(getdate()))  
  - AssignedByUserID: int (NOT NULL)  
  
PRIMARY KEY:  
  - IncidentCaseDoctorID  
  
FOREIGN KEYS:  
  - DoctorID -> APP_LOOKUP_DOCTOR.DoctorID  
  - IncidentRequestCaseID -> APP_IncidentCase.IncidentRequestCaseID  
  
  
============================================================  
TABLE: APP_IncidentCaseFeedback  
============================================================  
COLUMNS:  
  - IncidentRequestCaseID: int (NOT NULL)  
  - Cause_Staff_Training: bit  
  - Cause_Staff_Incentives: bit  
  - Cause_Staff_Competency: bit  
  - Cause_Staff_Understaffed: bit  
  - Cause_Staff_NonCompliance: bit  
  - Cause_Staff_NoCoordination: bit  
  - Cause_Staff_Other: bit  
  - Cause_Staff_OtherText: nvarchar(-1)  
  - Cause_Process_NotComprehensive: bit  
  - Cause_Process_Unclear: bit  
  - Cause_Process_MissingProtocol: bit  
  - Cause_Process_Other: bit  
  - Cause_Process_OtherText: nvarchar(-1)  
  - Cause_Equipment_NotAvailable: bit  
  - Cause_Equipment_SystemIncomplete: bit  
  - Cause_Equipment_HardToApply: bit  
  - Cause_Equipment_Other: bit  
  - Cause_Equipment_OtherText: nvarchar(-1)  
  - Cause_Environment_PlaceNature: bit  
  - Cause_Environment_Surroundings: bit  
  - Cause_Environment_WorkConditions: bit  
  - Cause_Environment_Other: bit  
  - Cause_Environment_OtherText: nvarchar(-1)  
  - Preventive_MonthlyMeetings: bit  
  - Preventive_TrainingPrograms: bit  
  - Preventive_IncreaseStaff: bit  
  - Preventive_MMCommitteeActions: bit  
  - Preventive_Other: bit  
  - Preventive_OtherText: nvarchar(-1)  
  - DepartmentExplanationText: nvarchar(-1)  
  - DepartmentExplanationStatusID: int (NOT NULL)  
  - DepartmentExplanationReceivalDate: date  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  - CreatedByUserID: int (NOT NULL)  
  
PRIMARY KEY:  
  - IncidentRequestCaseID  
  
FOREIGN KEYS:  
  - IncidentRequestCaseID -> APP_IncidentCase.IncidentRequestCaseID  
  
  
============================================================  
TABLE: APP_IncidentCaseTargetDepartment  
============================================================  
COLUMNS:  
  - TargetID: int (NOT NULL)  
  - IncidentRequestCaseID: int (NOT NULL)  
  - DepartmentID: int (NOT NULL)  
  - IsPrimary: bit (NOT NULL) (Default=((0)))  
  - AssignedAt: datetime (NOT NULL) (Default=(getdate()))  
  - AssignedByUserID: int (NOT NULL)  
  
PRIMARY KEY:  
  - TargetID  
  
FOREIGN KEYS:  
  - IncidentRequestCaseID -> APP_IncidentCase.IncidentRequestCaseID  
  
  
============================================================  
TABLE: APP_LOOKUP_CASE_STAGE  
============================================================  
COLUMNS:  
  - StageID: int (NOT NULL)  
  - StageName: nvarchar(100) (NOT NULL)  
  - StageOrder: int (NOT NULL)  
  
PRIMARY KEY:  
  - StageID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_CASE_STATUS  
============================================================  
COLUMNS:  
  - CaseStatusID: int (NOT NULL)  
  - Code: nvarchar(50) (NOT NULL)  
  - Name: nvarchar(200) (NOT NULL)  
  - IsFinal: bit (NOT NULL) (Default=((0)))  
  - IsActive: bit (NOT NULL) (Default=((1)))  
  - DisplayOrder: int (NOT NULL)  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  
PRIMARY KEY:  
  - CaseStatusID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_CATEGORY  
============================================================  
COLUMNS:  
  - CategoryID: int (NOT NULL)  
  - DomainID: int (NOT NULL)  
  - CategoryName: nvarchar(100) (NOT NULL)  
  - CategoryOrder: int (NOT NULL)  
  
PRIMARY KEY:  
  - CategoryID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_CLASSIFICATION  
============================================================  
COLUMNS:  
  - ClassificationID: int (NOT NULL)  
  - SubCategoryID: int (NOT NULL)  
  - Classification_AR: nvarchar(300) (NOT NULL)  
  - Classification_EN: nvarchar(300)  
  
PRIMARY KEY:  
  - ClassificationID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_CLINICAL_RISK_TYPE  
============================================================  
COLUMNS:  
  - ClinicalRiskTypeID: int (NOT NULL)  
  - Code: nvarchar(50) (NOT NULL)  
  - Name: nvarchar(200) (NOT NULL)  
  - IsActive: bit (NOT NULL) (Default=((1)))  
  - DisplayOrder: int (NOT NULL)  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  
PRIMARY KEY:  
  - ClinicalRiskTypeID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_DOCTOR  
============================================================  
COLUMNS:  
  - DoctorID: int (NOT NULL)  
  - DoctorName: nvarchar(200) (NOT NULL)  
  - Specialty: nvarchar(200)  
  - IsActive: bit (NOT NULL) (Default=((1)))  
  - SourceSystem: nvarchar(100)  
  - LastSyncedAt: datetime  
  
PRIMARY KEY:  
  - DoctorID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_DOMAIN  
============================================================  
COLUMNS:  
  - DomainID: int (NOT NULL)  
  - DomainCode: nvarchar(50) (NOT NULL)  
  - DomainName: nvarchar(100) (NOT NULL)  
  - DomainOrder: int (NOT NULL)  
  
PRIMARY KEY:  
  - DomainID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_EXPLANATION_STATUS  
============================================================  
COLUMNS:  
  - StatusID: int (NOT NULL)  
  - StatusName: nvarchar(50) (NOT NULL)  
  
PRIMARY KEY:  
  - StatusID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_FEEDBACK_INTENT_TYPE  
============================================================  
COLUMNS:  
  - FeedbackIntentTypeID: int (NOT NULL)  
  - Code: nvarchar(50) (NOT NULL)  
  - NameAr: nvarchar(200) (NOT NULL)  
  - NameEn: nvarchar(200) (NOT NULL)  
  - IsActive: bit (NOT NULL) (Default=((1)))  
  - DisplayOrder: int (NOT NULL)  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  
PRIMARY KEY:  
  - FeedbackIntentTypeID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_HARM_LEVEL  
============================================================  
COLUMNS:  
  - HarmID: int (NOT NULL)  
  - HarmLevel: nvarchar(50) (NOT NULL)  
  - SeverityOrder: int (NOT NULL)  
  
PRIMARY KEY:  
  - HarmID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_LOOKUP_SUBCATEGORY  
============================================================  
COLUMNS:  
  - SubCategoryID: int (NOT NULL)  
  - CategoryID: int (NOT NULL)  
  - SubCategoryName: nvarchar(150) (NOT NULL)  
  
PRIMARY KEY:  
  - SubCategoryID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: APP_OrgUnitPolicy  
============================================================  
COLUMNS:  
  - OrgUnitPolicyID: int (NOT NULL)  
  - OrgUnitID: int (NOT NULL)  
  - OrgUnitType: int (NOT NULL)  
  - LowSeverityLimit: int (NOT NULL)  
  - MediumSeverityLimit: int (NOT NULL)  
  - HighSeverityLimit: int (NOT NULL)  
  - ClinicalDomainLimit: int (NOT NULL)  
  - ManagementDomainLimit: int (NOT NULL)  
  - RelationalDomainLimit: int (NOT NULL)  
  - EnableLowSeverityRepetitionRule: bit (NOT NULL) (Default=((0)))  
  - EnableMediumSeverityRepetitionRule: bit (NOT NULL) (Default=((0)))  
  - EnableHighSeverityPercentageRule: bit (NOT NULL) (Default=((0)))  
  - EnableHighSeverityPercentageByDomainRule: bit (NOT NULL) (Default=((0)))  
  - IsActive: bit (NOT NULL) (Default=((1)))  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  - CreatedByUserID: int (NOT NULL)  
  - UpdatedAt: datetime  
  - UpdatedByUserID: int  
  
PRIMARY KEY:  
  - OrgUnitPolicyID  
  
FOREIGN KEYS:  
  - OrgUnitID -> AdminsrationUnit.UniqueID  
  
  
============================================================  
TABLE: APP_SeasonCase  
============================================================  
COLUMNS:  
  - SeasonCaseID: int (NOT NULL)  
  - SeasonID: int (NOT NULL)  
  - DepartmentID: int (NOT NULL)  
  - SeasonalReportText: nvarchar(-1)  
  - SeasonalReportDepartmentFeedback: nvarchar(-1)  
  - SeasonCaseStatusID: int (NOT NULL)  
  - CreatedAt: datetime (NOT NULL) (Default=(getdate()))  
  - CreatedByUserID: int (NOT NULL)  
  
PRIMARY KEY:  
  - SeasonCaseID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: IncidentRequest  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - YearCounter: int  
  - IncidentRequesterTypeID: int  
  - Code: nvarchar(50)  
  - PatientTypeID: int  
  - DoctorID: nvarchar(8)  
  - EmployeeID: int  
  - MRN: bigint  
  - SourceBuilding: nvarchar(3)  
  - PatientName: nvarchar(200)  
  - PatientID: bigint  
  - DateAndTimeRecieved: datetime (NOT NULL)  
  - IncidentStatusID: int  
  - IncidentSourceID: int  
  - SourceSectionID: int  
  - SourceDepartmentID: int  
  - SourceDepartmentName: nvarchar(100)  
  - SourceAdminID: int  
  - SourceAdminName: nvarchar(100)  
  - RequesterName: nvarchar(150)  
  - Note: nvarchar(1000)  
  - IsFeedbackRequested: bit  
  - IsFeedbackGiven: bit  
  - IsInPatient: bit  
  - DateAndTimeFeedbackGiven: datetime  
  - SatisfactoryID: int  
  - DateAndTimeCreated: datetime  
  - DateAndTimeUpdated: datetime  
  - CreatedByApplicationUserID: int  
  - UpdatedByApplicationUserID: int  
  - Frozen: bit (Default=((0)))  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  - SourceSectionID -> AdminsrationUnit.UniqueID  
  - IncidentRequesterTypeID -> Parameter.UniqueID  
  - IncidentStatusID -> Parameter.UniqueID  
  - IncidentSourceID -> Parameter.UniqueID  
  - SatisfactoryID -> Parameter.UniqueID  
  
  
============================================================  
TABLE: IncidentRequestCase  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - IncidentRequestID: int (NOT NULL)  
  - Description: nvarchar(1000)  
  - IncidentCaseCategoryID: int  
  - IncidentCaseSubCategoryID: int  
  - IncidentRequestCaseStatusID: int  
  - CaseBuilding: nvarchar(3)  
  - Note: nvarchar(-1)  
  - DoctorID: int  
  - DoctorSpecility: nvarchar(200)  
  - SectionID: int  
  - DepartmentID: int  
  - DeprtmentName: nvarchar(100)  
  - AdminID: int  
  - AdminName: nvarchar(100)  
  - DateAndTimeCreated: datetime  
  - DateAndTimeUpdated: datetime  
  - CreatedByApplicationUserID: int  
  - UpdatedByApplicationUserID: int  
  - DateAndTimeHappened: datetime  
  - IncidentTypeID: int  
  - Frozen: bit  
  - oldsectionID: int  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: IncidentRequestCaseAction  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - IncidentRequestCaseID: int (NOT NULL)  
  - Description: nvarchar(-1)  
  - IncidentRequestCaseActionMessageID: int  
  - IncidentRequestCaseActionInstructionID: int  
  - Note: nvarchar(-1)  
  - SectionNote: nvarchar(-1)  
  - DateAndTimeCreated: datetime  
  - DateAndTimeUpdated: datetime  
  - CreatedByApplicationUserID: int  
  - UpdatedByApplicationUserID: int  
  - IsAutoUpdated: bit  
  - SelectionNoteID: int  
  - SelectionNote: nvarchar(-1)  
  - JobDescriptionID: int  
  - JobDescriptionDetailID: int  
  - DateAndTimeSelectionNoteSaved: datetime  
  - SavedSelectionNoteByEmployeeID: int  
  - IsAutoSelectionNoteSaved: bit  
  - DateAndTimeSectionNoteSaved: datetime  
  - SavedSectionNoteByEmployeeID: int  
  - IsAutoSectionNoteSaved: bit  
  - DepartmentNote: nvarchar(-1)  
  - SavedDepartmentNoteByEmployeeID: int  
  - DateAndTimeDepartmentNoteSaved: datetime  
  - IsAutoDepartmentNoteSaved: bit  
  - SubmitByEmployeeID: int  
  - DateAndTimeSubmit: datetime  
  - Locked: bit (Default=((0)))  
  - ProblemReason: nvarchar(-1)  
  - ReasonNotResolvingID: int  
  - GoverningPolicies: nvarchar(-1)  
  - IsImprovementForm: bit  
  - Frozen: bit  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  - IncidentRequestCaseID -> IncidentRequestCase.UniqueID  
  - IncidentRequestCaseActionMessageID -> Parameter.UniqueID  
  - IncidentRequestCaseActionInstructionID -> Parameter.UniqueID  
  - ReasonNotResolvingID -> Parameter.UniqueID  
  - JobDescriptionID -> Parameter.UniqueID  
  - JobDescriptionDetailID -> Parameter.UniqueID  
  - SelectionNoteID -> Parameter.UniqueID  
  
  
============================================================  
TABLE: Instance  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - EnglishName: nvarchar(100) (NOT NULL)  
  - ParialEnglishName: nvarchar(100)  
  - ArabicName: nvarchar(100) (NOT NULL)  
  - ParialArabicName: nvarchar(100)  
  - DownloadLocation: nvarchar(1000) (NOT NULL)  
  - PublishLocation: nvarchar(1000) (NOT NULL)  
  - DamanContractNumber: int  
  - Address: nvarchar(200)  
  - Logo: image(2147483647)  
  - IsMain: bit  
  - CobolServerLinK: nvarchar(50)  
  - ArchiveFolderPath: nvarchar(100)  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: Parameter  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - Description: nvarchar(200)  
  - ParentID: int  
  - Frozen: bit (Default=((0)))  
  - CreateDate: datetime  
  - CreatedID: int  
  - Severity: int  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  - ParentID -> Parameter.UniqueID  
  
  
============================================================  
TABLE: Role  
============================================================  
COLUMNS:  
  - UniqueID: nchar(10) (NOT NULL)  
  - Name: nvarchar(50)  
  - IsActive: bit  
  
PRIMARY KEY:  
  None  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: Season  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - SeasonName: nvarchar(100)  
  - StartDate: date  
  - EndDate: date  
  - IsDone: bit (Default=((0)))  
  - Frozen: bit (Default=((0)))  
  - CreateDate: datetime  
  - CreateID: int  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: sysdiagrams  
============================================================  
COLUMNS:  
  - name: nvarchar(128) (NOT NULL)  
  - principal_id: int (NOT NULL)  
  - diagram_id: int (NOT NULL)  
  - version: int  
  - definition: varbinary(-1)  
  
PRIMARY KEY:  
  - diagram_id  
  
FOREIGN KEYS:  
  None  
  
  
============================================================  
TABLE: UserRole  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - UserID: int  
  - RoleID: int  
  - Frozen: bit (Default=((0)))  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  - RoleID -> Parameter.UniqueID  
  - UserID -> Users.UniqueID  
  
  
============================================================  
TABLE: Users  
============================================================  
COLUMNS:  
  - UniqueID: int (NOT NULL)  
  - Name: nvarchar(150)  
  - LoginName: nvarchar(50)  
  - Password: nvarchar(10)  
  - Frozen: bit (Default=((0)))  
  - CreatedBYID: int  
  - CreatedDate: datetime  
  - IsAdmin: bit  
  - IsIncidentUser: bit  
  - IsViewer: bit  
  - RoleID: int  
  - UserID: int  
  
PRIMARY KEY:  
  - UniqueID  
  
FOREIGN KEYS:  
  None