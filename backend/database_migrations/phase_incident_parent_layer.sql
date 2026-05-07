/*
Phase: Incident Parent Layer
Purpose:
  1) Create dbo.APP_Incident for shared/common fields.
  2) Add incident_id to dbo.APP_IncidentCase.
  3) Backfill existing cases with one incident per case.
  4) Add index + foreign key.

Notes:
  - Compatibility-first: existing case/subcase workflow remains.
  - Script is idempotent where practical.
*/

SET NOCOUNT ON;
SET XACT_ABORT ON;

BEGIN TRY
    BEGIN TRANSACTION;

    /* 1) Create parent table */
    IF OBJECT_ID(N'dbo.APP_Incident', N'U') IS NULL
    BEGIN
        CREATE TABLE dbo.APP_Incident (
            incident_id INT IDENTITY(1,1) NOT NULL PRIMARY KEY,
            incident_number AS (CONCAT('INC-', RIGHT(CONCAT('000000', incident_id), 6))) PERSISTED,

            /* Shared/common fields */
            patient_name NVARCHAR(255) NULL,
            primary_doctor_name NVARCHAR(255) NULL,
            primary_worker_name NVARCHAR(255) NULL,
            feedback_intent_type_id INT NULL,
            issuing_org_unit_id INT NULL,
            complaint_summary NVARCHAR(2000) NULL,
            building_id INT NULL,
            is_inpatient BIT NULL,

            created_at DATETIME2(0) NOT NULL CONSTRAINT DF_APP_Incident_created_at DEFAULT SYSUTCDATETIME(),
            created_by_user_id INT NULL,
            updated_at DATETIME2(0) NULL,
            updated_by_user_id INT NULL
        );
    END;

    /* 2) Add incident_id column to APP_IncidentCase */
    IF COL_LENGTH('dbo.APP_IncidentCase', 'incident_id') IS NULL
    BEGIN
        ALTER TABLE dbo.APP_IncidentCase
        ADD incident_id INT NULL;
    END;

    /* 3) Backfill strategy: one incident per existing case with null incident_id */
    IF OBJECT_ID(N'dbo.APP_IncidentCase', N'U') IS NOT NULL
       AND OBJECT_ID(N'dbo.APP_Incident', N'U') IS NOT NULL
    BEGIN
        IF OBJECT_ID('tempdb..#IncidentBackfillMap') IS NOT NULL
            DROP TABLE #IncidentBackfillMap;

        CREATE TABLE #IncidentBackfillMap (
            incident_id INT NOT NULL,
            IncidentRequestCaseID INT NOT NULL
        );

        ;WITH cases_to_backfill AS (
            SELECT
                c.IncidentRequestCaseID,
                c.PatientName,
                c.FeedbackIntentTypeID,
                c.IssuingOrgUnitID,
                c.ComplaintText,
                c.BuildingID,
                c.isINPatient,
                c.CreatedByUserID
            FROM dbo.APP_IncidentCase c
            WHERE c.incident_id IS NULL
        )
        INSERT INTO dbo.APP_Incident (
            patient_name,
            feedback_intent_type_id,
            issuing_org_unit_id,
            complaint_summary,
            building_id,
            is_inpatient,
            created_by_user_id
        )
        OUTPUT inserted.incident_id, src.IncidentRequestCaseID
        INTO #IncidentBackfillMap(incident_id, IncidentRequestCaseID)
        SELECT
            src.PatientName,
            src.FeedbackIntentTypeID,
            src.IssuingOrgUnitID,
            src.ComplaintText,
            src.BuildingID,
            src.isINPatient,
            src.CreatedByUserID
        FROM cases_to_backfill src;
    END;

    IF OBJECT_ID('tempdb..#IncidentBackfillMap') IS NOT NULL
    BEGIN
        UPDATE c
        SET c.incident_id = m.incident_id
        FROM dbo.APP_IncidentCase c
        INNER JOIN #IncidentBackfillMap m
            ON m.IncidentRequestCaseID = c.IncidentRequestCaseID
        WHERE c.incident_id IS NULL;
    END;

    /* 4) Index + FK (only if not already present) */
    IF NOT EXISTS (
        SELECT 1
        FROM sys.indexes
        WHERE object_id = OBJECT_ID(N'dbo.APP_IncidentCase')
          AND name = N'IX_APP_IncidentCase_incident_id'
    )
    BEGIN
        CREATE INDEX IX_APP_IncidentCase_incident_id
            ON dbo.APP_IncidentCase(incident_id);
    END;

    IF NOT EXISTS (
        SELECT 1
        FROM sys.foreign_keys
        WHERE name = N'FK_APP_IncidentCase_APP_Incident_incident_id'
    )
    BEGIN
        ALTER TABLE dbo.APP_IncidentCase
        WITH CHECK ADD CONSTRAINT FK_APP_IncidentCase_APP_Incident_incident_id
            FOREIGN KEY (incident_id)
            REFERENCES dbo.APP_Incident(incident_id);
    END;

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0
        ROLLBACK TRANSACTION;

    DECLARE @ErrMsg NVARCHAR(4000) = ERROR_MESSAGE();
    DECLARE @ErrSeverity INT = ERROR_SEVERITY();
    DECLARE @ErrState INT = ERROR_STATE();
    RAISERROR(@ErrMsg, @ErrSeverity, @ErrState);
END CATCH;
