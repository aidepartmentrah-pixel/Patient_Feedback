-- Confirms the migration-history baseline was recorded (011_record_database_version.sql).
SELECT MigrationID, MigrationName, AppliedAt, Success
FROM dbo.SchemaMigrationHistory
WHERE MigrationName = 'baseline_install_1.0.0';
-- Expect: exactly 1 row, Success = 1.
-- Note: on a FRESH install this should be the ONLY row. If this query returns
-- additional rows (phase_ml_s1_..., phase_ml_s8_...), you have restored an
-- EXISTING installation's data rather than run a fresh install -- see
-- ../scripts/restore_database.sql vs ../install/ for the distinction.
