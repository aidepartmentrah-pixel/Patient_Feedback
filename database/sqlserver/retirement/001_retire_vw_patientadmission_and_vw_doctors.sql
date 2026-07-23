-- ============================================================================
-- RETIREMENT MIGRATION -- REQUIRES MANUAL REVIEW AND APPROVAL BEFORE EXECUTION
-- Do NOT run this automatically as part of any install/update pipeline.
-- ============================================================================
-- CORRECTION (2026-07-23): this script originally also targeted VW_Doctors.
-- That was wrong — its justification ("doctor reads now go through
-- APP_LOOKUP_DOCTOR... VW_Doctors only appears in stale code comments") did
-- not match the actual code: backend/api/services/search_service.py's
-- search_doctors() (the function behind the incident-creation form's doctor
-- autocomplete) still ran a LIVE `SELECT ... FROM VW_Doctors` query, not a
-- comment. This is exactly the bug that broke doctor search on a fresh
-- install with no VW_Doctors data. VW_Doctors has now been removed from
-- this script's scope. It only becomes a real retirement candidate after
-- search_doctors() is migrated to the Hospital Directory API (Session C2,
-- landed this same commit — see api/services/staff_directory_service.py) —
-- write a separate, freshly dependency-checked retirement script for it
-- once that's been verified live, don't just re-add it here.
--
-- Removes VW_PatientAdmission only, confirmed obsolete 2026-07-21:
--   - patient reads moved to the Hospital Directory API;
--     patients_db.py no longer imports PATIENT_ADMISSION_TABLE at all.
-- Dependency check performed 2026-07-21 against live IncidentManager:
--   zero foreign keys reference this table (verified via sys.foreign_keys).
-- Before running against ANY installation:
--   1. Re-run the dependency check below and confirm zero rows returned.
--   2. Take a full database backup (see scripts/backup_database.sql).
--   3. Confirm no application code references this table (grep for
--      VW_PatientAdmission / PATIENT_ADMISSION_TABLE).

-- Step 1: Dependency re-check (run first, must return 0 rows before proceeding)
SELECT fk.name AS fk_name, tp.name AS parent_table, tr.name AS ref_table
FROM sys.foreign_keys fk
JOIN sys.tables tp ON fk.parent_object_id = tp.object_id
JOIN sys.tables tr ON fk.referenced_object_id = tr.object_id
WHERE tp.name IN ('VW_PatientAdmission')
   OR tr.name IN ('VW_PatientAdmission');

-- Step 2: Existence-checked drop (only run after Step 1 confirms zero rows,
-- and after a backup has been taken)
-- IF OBJECT_ID('dbo.VW_PatientAdmission', 'U') IS NOT NULL
--     DROP TABLE dbo.VW_PatientAdmission;

-- Step 3: Record the retirement in SchemaMigrationHistory once executed
-- INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success)
-- VALUES ('retire_obsolete_hospital_view_tables', NULL, SYSUTCDATETIME(), SUSER_SNAME(), '1.0.0', 1);
