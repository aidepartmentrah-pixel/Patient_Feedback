# Database Install Guide

For creating a brand-new `IncidentManager` database — Docker dev environment on the Legion,
a validation environment, or eventually a real offline deployment. **Never run this against an
installation that already has real data** — see `DATABASE_UPDATE_GUIDE.md` for that case instead.

## Prerequisites

- A SQL Server instance reachable from wherever the scripts run (Docker container, local
  install, etc.) with a login that has `db_owner` (or at minimum `db_ddladmin` +
  `db_datareader` + `db_datawriter`, matching this project's existing `HCAT_Insight` login).
- An empty target database — these scripts do not create the database itself
  (`001_create_database.sql` is a placeholder for that step; SQL Server Docker images typically
  handle database creation separately via `CREATE DATABASE` before running this package).

## Run order

Execute `install/*.sql` in numeric order — each one is idempotent (`IF NOT EXISTS`/
`IF OBJECT_ID(...) IS NULL` guards throughout), so re-running after a partial failure is safe.

1. `001_create_database.sql` — placeholder; create the database itself first if your deployment
   tooling doesn't already do this (e.g. the SQL Server Docker init service in Step 2A).
2. `002_create_schema.sql` — all 81 tables (creates the `ml` schema too). Does **not** create
   `VW_PatientAdmission`/`VW_Doctors` — confirmed obsolete, see `DATABASE_STRUCTURE_REPORT.md`.
3. `003_create_indexes.sql` — non-PK indexes.
4. `004_create_constraints.sql` — primary keys and foreign keys.
5. `005_create_views.sql`, `006_create_stored_procedures.sql`, `007_create_triggers.sql` —
   intentionally empty; none exist in the live database.
6. `008_seed_lookup_data.sql` — 19 universal lookup tables (classification categories,
   severity levels, harm levels, statuses, etc.) plus `ml.EmbeddingModelVersion`.
7. `009_seed_configuration.sql` — `APP_OrgUnitPolicy`, `APP_DepartmentPolicy`,
   `APP_DepartmentEvaluationRule`, `APP_Roles`. Org-structure/policy data, not patient data.
8. `010_seed_users_roles.sql` — pointer file only; role definitions moved into step 7. No user
   accounts are ever seeded — see "After installing" below.
9. `011_record_database_version.sql` — records `baseline_install_1.0.0` in
   `dbo.SchemaMigrationHistory`.

## After installing

- Run `../validation/*.sql` to confirm the install completed correctly (see
  `DATABASE_VALIDATION_GUIDE.md`).
- Create user accounts through the application's own bootstrap/admin flow —
  `validate_users.sql` explicitly expects `dbo.APP_Users` to be empty right after install.
- Point the backend's `config/db_settings.json` (or env var overrides) at the new database and
  start the app.

## What you will NOT get from a fresh install

- No patient/complaint records, no `APP_LOOKUP_DOCTOR` sync-cache rows, no `ml.CaseTrainingRecord`
  or `ml.HistoricalTrainingExample` rows (real complaint text + embeddings — explicitly excluded,
  see `DATABASE_STRUCTURE_REPORT.md`), no user accounts. If you need an installation's actual
  historical data, use `DATABASE_RESTORE_GUIDE.md` instead of this install path.
