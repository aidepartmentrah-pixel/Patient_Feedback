# Database Restore Guide

Restores a backup taken via `DATABASE_BACKUP_GUIDE.md` onto a target SQL Server instance —
used when moving an existing installation's real data to another server (e.g. a genuine
migration to new hardware), not for provisioning a new environment from scratch.

## Steps

1. Copy the `.bak` file to somewhere the target SQL Server instance/container can read it.
2. Open `sqlserver/scripts/restore_database.sql`.
3. Run the commented-out `RESTORE FILELISTONLY FROM DISK = @BackupPath;` first, and compare the
   `LogicalName` values it returns against the `MOVE 'IncidentManager' TO ...` /
   `MOVE 'IncidentManager_log' TO ...` lines in the script — these will differ if the backup came
   from a differently-configured SQL Server instance (different logical file names).
4. Update `@BackupPath` and `@DataPath` for the target environment, and the `MOVE` logical names
   if step 3 showed they differ.
5. Run the `RESTORE DATABASE` statement.
6. Run `sqlserver/scripts/verify_database.sql` — `DBCC CHECKDB`, sanity row counts on tables that
   should never be empty on a real installation, and a dump of `dbo.SchemaMigrationHistory` to
   confirm you restored a real installation's history (more than just the `baseline_install_1.0.0`
   row a fresh install would have).

## After restoring

The application's `config/db_settings.json` (or env var overrides) needs to point at wherever
the restore landed. No further seeding is needed or wanted — a restore already carries the full
real installation, including everything `install/` deliberately leaves out.
