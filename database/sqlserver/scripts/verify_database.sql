-- ============================================================================
-- Post-restore integrity check -- run after restore_database.sql to confirm
-- the restored installation's real data is intact. (For verifying a FRESH
-- INSTALL instead, see ../validation/ -- different purpose: this script
-- checks an existing installation's data came through a backup/restore
-- cycle intact; validation/ checks a fresh install's schema+seed is complete.)
-- ============================================================================
DECLARE @DBName SYSNAME = N'IncidentManager';

-- Physical/logical consistency check
DBCC CHECKDB (N'IncidentManager') WITH NO_INFOMSGS, ALL_ERRORMSGS;

-- Row-count sanity check on a sample of tables that should never be empty
-- on a real installation (adjust/expand as needed -- this is a smoke test,
-- not exhaustive)
SELECT 'APP_Users' AS table_name, COUNT(*) AS row_count FROM dbo.APP_Users
UNION ALL SELECT 'APP_IncidentCase', COUNT(*) FROM dbo.APP_IncidentCase
UNION ALL SELECT 'APP_LOOKUP_CLASSIFICATION', COUNT(*) FROM dbo.APP_LOOKUP_CLASSIFICATION
UNION ALL SELECT 'SchemaMigrationHistory', COUNT(*) FROM dbo.SchemaMigrationHistory
UNION ALL SELECT 'ml.HistoricalTrainingExample', COUNT(*) FROM ml.HistoricalTrainingExample
UNION ALL SELECT 'ml.CaseTrainingRecord', COUNT(*) FROM ml.CaseTrainingRecord;

-- Confirm the migration history table reflects what this specific
-- installation has actually applied (should NOT be empty on a restored
-- production copy, unlike a fresh install which only has the baseline row)
SELECT MigrationID, MigrationName, AppliedAt, Success FROM dbo.SchemaMigrationHistory ORDER BY MigrationID;
