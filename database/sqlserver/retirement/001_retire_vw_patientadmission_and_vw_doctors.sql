-- ============================================================================
-- RETIREMENT MIGRATION -- REQUIRES MANUAL REVIEW AND APPROVAL BEFORE EXECUTION
-- Do NOT run this automatically as part of any install/update pipeline.
-- ============================================================================
-- Removes VW_PatientAdmission and VW_Doctors, confirmed obsolete 2026-07-21:
--   - VW_PatientAdmission: patient reads moved to the Hospital Directory API;
--     patients_db.py no longer imports PATIENT_ADMISSION_TABLE at all.
--   - VW_Doctors: doctor reads now go through APP_LOOKUP_DOCTOR (API-synced
--     cache table); VW_Doctors only appears in stale code comments, never
--     in a live query.
-- Dependency check performed 2026-07-21 against live IncidentManager:
--   zero foreign keys reference either table (verified via sys.foreign_keys).
-- Before running against ANY installation:
--   1. Re-run the dependency check below and confirm zero rows returned.
--   2. Take a full database backup (see scripts/backup_database.sql).
--   3. Confirm no application code references these tables (grep for
--      VW_PatientAdmission / VW_Doctors / PATIENT_ADMISSION_TABLE / DOCTORS_TABLE).

-- Step 1: Dependency re-check (run first, must return 0 rows before proceeding)
SELECT fk.name AS fk_name, tp.name AS parent_table, tr.name AS ref_table
FROM sys.foreign_keys fk
JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
WHERE tp.name IN ('VW_PatientAdmission', 'VW_Doctors')
   OR tr.name IN ('VW_PatientAdmission', 'VW_Doctors');

-- Step 2: Existence-checked drops (only run after Step 1 confirms zero rows,
-- and after a backup has been taken)
-- IF OBJECT_ID('dbo.VW_PatientAdmission', 'U') IS NOT NULL
--     DROP TABLE dbo.VW_PatientAdmission;
-- IF OBJECT_ID('dbo.VW_Doctors', 'U') IS NOT NULL
--     DROP TABLE dbo.VW_Doctors;

-- Step 3: Record the retirement in SchemaMigrationHistory once executed
-- INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
-- VALUES ('retire_obsolete_hospital_view_tables', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.0.0', 1);
