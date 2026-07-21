-- Phase: Decision Propagation to All Levels
-- Adds a per-level acknowledgment table so PATIENT_SERVICES_DECISION_COMPLETED
-- stays visible to every org level that historically handled the case
-- (Section/Department/Administration) until THAT level acknowledges it,
-- instead of the single subcase Status flip (which previously removed the
-- item from every level's inbox the instant any one of them acknowledged it).
-- Run once against the live DB, then restart the backend.

IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'dbo' AND TABLE_NAME = 'APP_SubcaseDecisionAcknowledgment'
)
BEGIN
    CREATE TABLE dbo.APP_SubcaseDecisionAcknowledgment (
        AcknowledgmentID     INT IDENTITY(1,1) PRIMARY KEY,
        SubcaseID            INT           NOT NULL,
        OrgLevel             VARCHAR(20)   NOT NULL,  -- 'section' | 'department' | 'administration'
        AcknowledgedByUserID INT           NOT NULL,
        AcknowledgedAt       DATETIME      NOT NULL DEFAULT GETDATE(),
        CONSTRAINT UQ_SubcaseDecisionAck UNIQUE (SubcaseID, OrgLevel),
        CONSTRAINT FK_SubcaseDecisionAck_Subcase FOREIGN KEY (SubcaseID)
            REFERENCES dbo.APP_AdministrativeSubcase(SubcaseID)
    )
    PRINT 'Created APP_SubcaseDecisionAcknowledgment.'
END
ELSE
BEGIN
    PRINT 'APP_SubcaseDecisionAcknowledgment already exists, skipping.'
END
