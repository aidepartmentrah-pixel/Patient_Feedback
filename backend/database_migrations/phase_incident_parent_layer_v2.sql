/*
Phase: Incident Parent Layer (v2 — fixes column parse-time error)
Uses dynamic SQL for the backfill so SQL Server does not validate
incident_id before ALTER TABLE has created it.
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
        PRINT 'Created dbo.APP_Incident';
    END
    ELSE
        PRINT 'dbo.APP_Incident already exists — skipped';

    /* 2) Add incident_id column to APP_IncidentCase */
    IF COL_LENGTH('dbo.APP_IncidentCase', 'incident_id') IS NULL
    BEGIN
        ALTER TABLE dbo.APP_IncidentCase ADD incident_id INT NULL;
        PRINT 'Added incident_id column to APP_IncidentCase';
    END
    ELSE
        PRINT 'incident_id column already exists — skipped';

    /* 3) Backfill — wrapped in dynamic SQL so column reference is resolved at runtime */
    EXEC sp_executesql N'
        IF OBJECT_ID(''tempdb..#IncidentBackfillMap'') IS NOT NULL
            DROP TABLE #IncidentBackfillMap;

        CREATE TABLE #IncidentBackfillMap (
            incident_id INT NOT NULL,
            IncidentRequestCaseID INT NOT NULL
        );

        INSERT INTO dbo.APP_Incident (
            patient_name, feedback_intent_type_id, issuing_org_unit_id,
            complaint_summary, building_id, is_inpatient, created_by_user_id
        )
        OUTPUT inserted.incident_id, src.IncidentRequestCaseID
        INTO #IncidentBackfillMap(incident_id, IncidentRequestCaseID)
        SELECT
            c.PatientName,
            c.FeedbackIntentTypeID,
            c.IssuingOrgUnitID,
            c.ComplaintText,
            c.BuildingID,
            c.isINPatient,
            c.CreatedByUserID
        FROM dbo.APP_IncidentCase c
        WHERE c.incident_id IS NULL;

        UPDATE c
        SET c.incident_id = m.incident_id
        FROM dbo.APP_IncidentCase c
        INNER JOIN #IncidentBackfillMap m ON m.IncidentRequestCaseID = c.IncidentRequestCaseID
        WHERE c.incident_id IS NULL;

        PRINT ''Backfilled cases with incident records'';
    ';

    /* 4) Index */
    IF NOT EXISTS (
        SELECT 1 FROM sys.indexes
        WHERE object_id = OBJECT_ID(N'dbo.APP_IncidentCase')
          AND name = N'IX_APP_IncidentCase_incident_id'
    )
    BEGIN
        CREATE INDEX IX_APP_IncidentCase_incident_id ON dbo.APP_IncidentCase(incident_id);
        PRINT 'Created index IX_APP_IncidentCase_incident_id';
    END;

    /* 5) Foreign key */
    IF NOT EXISTS (
        SELECT 1 FROM sys.foreign_keys
        WHERE name = N'FK_APP_IncidentCase_APP_Incident_incident_id'
    )
    BEGIN
        ALTER TABLE dbo.APP_IncidentCase
        WITH CHECK ADD CONSTRAINT FK_APP_IncidentCase_APP_Incident_incident_id
            FOREIGN KEY (incident_id) REFERENCES dbo.APP_Incident(incident_id);
        PRINT 'Added FK_APP_IncidentCase_APP_Incident_incident_id';
    END;

    COMMIT TRANSACTION;
    PRINT 'Migration completed successfully.';
END TRY
BEGIN CATCH
    IF @@TRANCOUNT > 0 ROLLBACK TRANSACTION;
    DECLARE @ErrMsg NVARCHAR(4000) = ERROR_MESSAGE();
    DECLARE @ErrSev INT = ERROR_SEVERITY();
    DECLARE @ErrSt INT = ERROR_STATE();
    RAISERROR(@ErrMsg, @ErrSev, @ErrSt);
END CATCH;
