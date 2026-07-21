-- ============================================================================
-- Full backup of an EXISTING IncidentManager installation's real data.
-- This is the mechanism for moving a specific hospital's installation
-- (including its real ML training history, patient reserve data, incidents,
-- etc.) to another server -- NOT the same thing as the install/ package,
-- which only ever creates schema + universal/config seed data.
-- ============================================================================
-- Replace @BackupPath before running. Use a path outside the SQL Server
-- container/service data directory so the backup survives container/service
-- replacement.

-- Example: N'C:\SQLBackup\IncidentManager_2026-07-21.bak' -- pick an actual date/tag, T-SQL does not expand variables in the filename for you
DECLARE @BackupPath NVARCHAR(500) = N'C:\SQLBackup\IncidentManager_REPLACE_ME.bak';
DECLARE @DBName SYSNAME = N'IncidentManager';

BACKUP DATABASE @DBName
TO DISK = @BackupPath
WITH FORMAT, INIT, COMPRESSION,
     NAME = N'IncidentManager-Full',
     STATS = 10;

-- Verify the backup is readable immediately after taking it
RESTORE VERIFYONLY FROM DISK = @BackupPath;
