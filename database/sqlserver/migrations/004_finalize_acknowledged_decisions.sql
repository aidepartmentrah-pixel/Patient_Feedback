-- ================================================================
-- MIGRATION: 004_finalize_acknowledged_decisions
-- ================================================================
-- Purpose: One-time data fix. A completed Patient Services (HCAT)
--          decision used to be finalized (DECISION_ACKNOWLEDGED) only
--          after section, department AND administration all clicked
--          OK. A section attached directly to an Administration has no
--          Department user, so those cases stayed
--          PATIENT_SERVICES_DECISION_COMPLETED forever.
--
--          The backend now requires only the levels that exist for the
--          target unit (case_creation_service.get_applicable_workflow_
--          levels). This script applies the same rule to cases that are
--          already stuck: every level that exists has acknowledged ->
--          DECISION_ACKNOWLEDGED. Nothing else is changed.
--
--          Levels per target unit type (same as the backend):
--            Administration (323)                  -> administration
--            Department (325)                      -> department, administration
--            Section (324) under Department (325)  -> section, department, administration
--            Section (324) under Administration    -> section, administration
--
-- Safe to re-run. Run it against the application database (the
-- one the backend's DB_DATABASE points at).
--
-- Date: 2026-09-23
-- ================================================================

;WITH required AS (
    SELECT s.SubcaseID, lvl.OrgLevel
    FROM dbo.APP_AdministrativeSubcase s
    JOIN dbo.AdminsrationUnit t ON t.UniqueID = s.TargetOrgUnitID
    LEFT JOIN dbo.AdminsrationUnit p ON p.UniqueID = t.ParentID
    CROSS APPLY (VALUES ('section'), ('department'), ('administration')) AS lvl(OrgLevel)
    WHERE s.Status = 'PATIENT_SERVICES_DECISION_COMPLETED'
      AND (
            (lvl.OrgLevel = 'administration' AND t.Type IN (323, 325, 324))
         OR (lvl.OrgLevel = 'department'     AND (t.Type = 325 OR (t.Type = 324 AND p.Type = 325)))
         OR (lvl.OrgLevel = 'section'        AND t.Type = 324)
      )
),
fully_acknowledged AS (
    SELECT r.SubcaseID
    FROM required r
    LEFT JOIN dbo.APP_SubcaseDecisionAcknowledgment a
           ON a.SubcaseID = r.SubcaseID AND a.OrgLevel = r.OrgLevel
    GROUP BY r.SubcaseID
    -- every required level has at least one ack row
    HAVING COUNT(DISTINCT CASE WHEN a.AcknowledgmentID IS NOT NULL THEN r.OrgLevel END)
         = COUNT(DISTINCT r.OrgLevel)
)
UPDATE s
SET Status    = 'DECISION_ACKNOWLEDGED',
    UpdatedAt = GETDATE()
FROM dbo.APP_AdministrativeSubcase s
JOIN fully_acknowledged f ON f.SubcaseID = s.SubcaseID
WHERE s.Status = 'PATIENT_SERVICES_DECISION_COMPLETED';

PRINT CONCAT(@@ROWCOUNT, ' stuck decision(s) finalized');
GO

IF NOT EXISTS (SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = '004_finalize_acknowledged_decisions')
    INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
    VALUES ('004_finalize_acknowledged_decisions', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.2.1', 1);
GO
