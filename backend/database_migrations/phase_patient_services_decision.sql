-- Phase: Patient Services Scientific Decision
-- Adds two new subcase status codes and four new columns on APP_AdministrativeSubcase.
-- Run once against the live DB, then restart the backend.

-- ---------------------------------------------------------------
-- 1. New status codes
-- ---------------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus
    WHERE StatusCode = 'WAITING_PATIENT_SERVICES_DECISION'
)
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus
        (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES (
        'WAITING_PATIENT_SERVICES_DECISION',
        'Waiting Patient Services Decision',
        N'بانتظار قرار خدمات المرضى بحسب المراجع العلميّة',
        12, 0, 1
    )
END

IF NOT EXISTS (
    SELECT 1 FROM dbo.APP_Lookup_SubcaseStatus
    WHERE StatusCode = 'PATIENT_SERVICES_DECISION_COMPLETED'
)
BEGIN
    INSERT INTO dbo.APP_Lookup_SubcaseStatus
        (StatusCode, StatusNameEn, StatusNameAr, DisplayOrder, IsFinal, IsActive)
    VALUES (
        'PATIENT_SERVICES_DECISION_COMPLETED',
        'Patient Services Decision Completed',
        N'تم إدخال قرار خدمات المرضى بحسب المراجع العلميّة',
        13, 0, 1
    )
END

-- ---------------------------------------------------------------
-- 2. New columns on APP_AdministrativeSubcase (all nullable)
-- ---------------------------------------------------------------
IF NOT EXISTS (
    SELECT 1 FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_SCHEMA = 'dbo'
      AND TABLE_NAME   = 'APP_AdministrativeSubcase'
      AND COLUMN_NAME  = 'PatientServicesDecisionText'
)
BEGIN
    ALTER TABLE dbo.APP_AdministrativeSubcase
        ADD PatientServicesDecisionText      NVARCHAR(MAX) NULL,
            PatientServicesDecisionByUserID  INT           NULL,
            PatientServicesDecisionAt        DATETIME      NULL,
            PatientServicesDecisionUpdatedAt DATETIME      NULL
END
