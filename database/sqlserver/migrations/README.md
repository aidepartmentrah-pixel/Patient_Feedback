# Schema Migrations

This folder holds **future schema-evolution scripts** for `IncidentManager` — the mechanism for
changing an *already-installed* database without recreating it or losing data.

## The existing convention (do not replace it)

Production already has a working migration-tracking table, `dbo.SchemaMigrationHistory`,
created before this database package existed. Its columns:

| Column | Purpose |
|---|---|
| `MigrationID` | identity PK |
| `MigrationName` | unique name of the migration (matches this folder's filenames minus `.sql`) |
| `Checksum` | optional, for detecting a migration file changed after being applied |
| `AppliedAt` | `datetime2`, when it ran |
| `AppliedBy` | `SUSER_SNAME()` of whoever ran it |
| `ApplicationVersion` | optional, which app release this shipped with |
| `Success` | bit |

Two migrations are already recorded from before this package existed:
`phase_ml_s1_create_ml_schema_and_tables` and `phase_ml_s8_historical_migration_schema`
(both applied 2026-07-16). This package's `install/011_record_database_version.sql` adds a
`baseline_install_1.0.0` row on top of that history for fresh installs.

## Writing a new migration

1. Name the file `NNN_short_description.sql` (numeric prefix, ascending).
2. Wrap the actual DDL in an idempotency guard (`IF NOT EXISTS (...) ... ` / `IF OBJECT_ID(...) IS NULL`),
   matching the style already used in `install/002_create_schema.sql`.
3. End the file with an insert into `SchemaMigrationHistory` using the same `MigrationName` as the
   filename, so `AppliedAt`/`Success` gets recorded the same way the two pre-existing migrations were.
4. Migrations must **never** `DROP` or destructively alter a table that might hold real hospital data
   without a corresponding entry in `../retirement/` and explicit sign-off — see that folder's README.

## What does NOT belong here

- Seeding lookup/config data — that's `install/008_seed_lookup_data.sql` and `009_seed_configuration.sql`
  (idempotent, safe to re-run, but not tracked as a "migration" since they don't change structure).
- Removing obsolete objects — that's `../retirement/`, which requires manual review before execution
  (see `../retirement/README.md`).
- Moving an existing installation's actual data to another server — that's backup/restore
  (`../scripts/backup_database.sql`, `../scripts/restore_database.sql`), not a migration.
