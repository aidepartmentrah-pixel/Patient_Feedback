-- ============================================================================
-- Restores a full IncidentManager backup (see backup_database.sql) onto a
-- target SQL Server instance. Used to move an EXISTING installation's real
-- data to another server -- not part of the fresh-install pipeline.
-- ============================================================================
-- Replace @BackupPath and the physical file paths below before running.
-- Run RESTORE FILELISTONLY first to confirm the logical file names match
-- what this script assumes (they will differ if the backup came from a
-- differently-configured instance).

DECLARE @BackupPath NVARCHAR(500) = N'C:\SQLBackup\IncidentManager_REPLACE_ME.bak';
DECLARE @DBName SYSNAME = N'IncidentManager';
DECLARE @DataPath NVARCHAR(500) = N'C:\Program Files\Microsoft SQL Server\MSSQL16.SQLEXPRESS\MSSQL\DATA\';

-- Step 1: inspect the backup's logical file names before restoring
-- RESTORE FILELISTONLY FROM DISK = @BackupPath;

-- Step 2: restore -- adjust logical file names (LogicalName column from
-- Step 1's output) if they differ from 'IncidentManager' / 'IncidentManager_log'.
-- RESTORE ... MOVE ... TO requires string literals, not variables/expressions,
-- so this builds and executes the statement as dynamic SQL.
DECLARE @RestoreSql NVARCHAR(MAX) = N'
RESTORE DATABASE ' + QUOTENAME(@DBName) + N'
FROM DISK = ''' + REPLACE(@BackupPath, '''', '''''') + N'''
WITH MOVE ''IncidentManager'' TO ''' + REPLACE(@DataPath, '''', '''''') + N'IncidentManager.mdf'',
     MOVE ''IncidentManager_log'' TO ''' + REPLACE(@DataPath, '''', '''''') + N'IncidentManager_log.ldf'',
     REPLACE,
     STATS = 10;';

EXEC sp_executesql @RestoreSql;

-- Step 3: sanity check after restore (run from master; RESTORE leaves session context on master)
SELECT name, state_desc, create_date FROM sys.databases WHERE name = @DBName;
