# Database Backup Guide

For moving or protecting an **existing installation's real data** — not part of the fresh-install
path. See `DATABASE_UPDATE_GUIDE.md` if you're unsure which mechanism you need.

## Running a backup

1. Open `sqlserver/scripts/backup_database.sql`.
2. Replace `@BackupPath` with an actual path — **outside** the SQL Server data directory (and
   outside any Docker container's ephemeral filesystem, once this project is containerized) so
   the backup survives a container replacement or service reinstall.
3. Run it. It performs a full `BACKUP DATABASE ... WITH COMPRESSION` and immediately runs
   `RESTORE VERIFYONLY` against the result to confirm the file isn't corrupt.

## When to back up

Per the RAH Lab Database Backup Policy — before: initial production deployment, any database
migration (`migrations/` or `retirement/`), application updates, infrastructure changes
affecting database services.

## What's in a backup

Everything — the full 83-table database including `ml.HistoricalTrainingExample`,
`ml.CaseTrainingRecord`, `APP_Users`, real incident/complaint records, everything the fresh-install
seed scripts deliberately exclude. That's the point: a backup is this specific installation's
complete state, not a portable release artifact.

## Restoring

See `DATABASE_RESTORE_GUIDE.md`.
