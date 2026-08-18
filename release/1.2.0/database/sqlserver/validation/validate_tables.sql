-- Confirms every table this package expects to exist actually does,
-- and that obsolete tables are NOT present (fresh install should never create them).

-- Expected tables (fails/returns rows if missing):

;WITH Expected(SchemaName, TableName) AS (
    SELECT * FROM (VALUES ('dbo','APP_LOOKUP_BUILDING'), ('dbo','APP_LOOKUP_CASE_STAGE'), ('dbo','APP_LOOKUP_CASE_STATUS'), ('dbo','APP_LOOKUP_CATEGORY'), ('dbo','APP_LOOKUP_CLASSIFICATION'), ('dbo','APP_LOOKUP_CLINICAL_RISK_TYPE'), ('dbo','APP_LOOKUP_DOMAIN'), ('dbo','APP_LOOKUP_EXPLANATION_STATUS'), ('dbo','APP_LOOKUP_FEEDBACK_INTENT_TYPE'), ('dbo','APP_LOOKUP_HARM_LEVEL'), ('dbo','APP_LOOKUP_RECORD_TYPE'), ('dbo','APP_Lookup_SatisfactionStatus'), ('dbo','APP_LOOKUP_SEVERITY'), ('dbo','APP_LOOKUP_SOURCE'), ('dbo','APP_Lookup_SubcaseActionItemStatus'), ('dbo','APP_Lookup_SubcaseStatus'), ('dbo','APP_Lookup_SubcaseType'), ('dbo','APP_LOOKUP_SUBCATEGORY'), ('ml','EmbeddingModelVersion'), ('dbo','APP_OrgUnitPolicy'), ('dbo','APP_DepartmentPolicy'), ('dbo','APP_DepartmentEvaluationRule'), ('dbo','APP_Roles'), ('ml','CaseTrainingRecord'), ('ml','HistoricalTrainingExample'), ('ml','EmbeddingProcessingJob'), ('ml','ImportBatch'), ('ml','ImportSourceRecordMap'), ('ml','LegacyDbSizeHistory'), ('ml','LegacyModelMetricHistory'), ('ml','LegacyTrainingRunHistory'), ('dbo','SchemaMigrationHistory'), ('dbo','APP_RCAFactorCategory'), ('dbo','APP_RCASuggestion'), ('dbo','APP_DrawerLabel'), ('dbo','Instance'), ('dbo','Parameter'), ('dbo','Role'), ('dbo','APP_LOOKUP_DOCTOR'), ('dbo','VW_HrEmployeeProfileView'), ('dbo','AdminsrationUnit'), ('dbo','AdminsrationUnitHistory'), ('dbo','APP_ActionItem'), ('dbo','APP_AdministrativeSubcase'), ('dbo','APP_CUSTOM_VIEWS'), ('dbo','APP_DataMigration_Map'), ('dbo','APP_DrawerNote'), ('dbo','APP_DrawerNoteLabelLink'), ('dbo','APP_ExternalApiSettings'), ('dbo','APP_HardwareConfig'), ('dbo','APP_Incident'), ('dbo','APP_IncidentCase'), ('dbo','APP_IncidentCaseDoctor'), ('dbo','APP_IncidentCaseEmployee'), ('dbo','APP_IncidentCaseFeedback'), ('dbo','APP_IncidentCaseSatisfaction'), ('dbo','APP_IncidentCaseTargetDepartment'), ('dbo','APP_PublicationBatch'), ('dbo','APP_PublicationBatchCase'), ('dbo','APP_ReportConfig'), ('dbo','APP_RESERVE_DOCTOR'), ('dbo','APP_RESERVE_PATIENT'), ('dbo','APP_SeasonalOrgUnitReport'), ('dbo','APP_SeasonalOrgUnitReport_ClassificationStats'), ('dbo','APP_SeasonalOrgUnitReport_PolicySnapshot'), ('dbo','APP_SeasonCase'), ('dbo','APP_SubcaseActionItem'), ('dbo','APP_SubcaseActionItemChangeNotice'), ('dbo','APP_SubcaseDecisionAcknowledgment'), ('dbo','APP_SubcaseRCASuggestionSelection'), ('dbo','APP_SupervisorActionItem'), ('dbo','APP_SupervisorActionItemAuditLog'), ('dbo','APP_SystemSettings'), ('dbo','APP_UserRoleScope'), ('dbo','APP_Users'), ('dbo','IncidentRequest'), ('dbo','IncidentRequestCase'), ('dbo','IncidentRequestCaseAction'), ('dbo','Season'), ('dbo','UserRole'), ('dbo','Users')) AS v(SchemaName, TableName)
)
SELECT e.SchemaName, e.TableName AS MissingTable
FROM Expected e
LEFT JOIN INFORMATION_SCHEMA.TABLES t
    ON t.TABLE_SCHEMA = e.SchemaName AND t.TABLE_NAME = e.TableName AND t.TABLE_TYPE = 'BASE TABLE'
WHERE t.TABLE_NAME IS NULL;
-- Expect: 0 rows. Any row means a table this package should have created is missing.

-- Obsolete tables that should NOT exist on a fresh install (fails/returns rows if present):

SELECT TABLE_NAME AS UnexpectedObsoleteTable
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_NAME IN ('VW_PatientAdmission', 'VW_Doctors');
-- Expect: 0 rows on a fresh install. Rows here on an EXISTING installation are
-- expected until the retirement migration (../retirement/) is reviewed and run.
