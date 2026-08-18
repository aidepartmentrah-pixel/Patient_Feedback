# Database Validation Guide

Post-install structural checks — confirms `install/` actually produced the expected end-state.
This checks *structure*, not full application behavior; see the note at the bottom for the
stronger, execution-based test that was proposed but not run during package development.

## Scripts (`sqlserver/validation/`)

Run each and compare against the expectations noted inline:

| Script | Checks | Expect |
|---|---|---|
| `validate_tables.sql` | All 81 expected tables exist; the 2 obsolete tables do not | 0 rows both queries (fresh install) |
| `validate_lookup_data.sql` | All 19 universal lookup tables were seeded | every `row_count` > 0 |
| `validate_configuration.sql` | `APP_OrgUnitPolicy`/`APP_DepartmentPolicy`/`APP_DepartmentEvaluationRule`/`APP_Roles` were seeded | row counts match `DATABASE_STRUCTURE_REPORT.md` baseline |
| `validate_constraints.sql` | PK/FK/index counts | matches the baseline noted in the script |
| `validate_users.sql` | Role definitions seeded, user accounts NOT seeded | `role_definition_count` > 0, `user_account_count` = 0 |
| `validate_database_version.sql` | `baseline_install_1.0.0` recorded in `SchemaMigrationHistory` | exactly 1 row, `Success = 1`, and it's the ONLY row on a fresh install |

## What this does not prove

These scripts confirm the install produced the right *shape* — they do not prove the
application can actually run against the result. A stronger test (proposed but not executed
during this package's development, at the user's discretion): create a scratch database, run
`install/` against it for real, then point a second temporary backend process at it and
exercise a few real endpoints. That catches gaps these structural checks can't — e.g. a column
type mismatch that only surfaces when the ORM/query layer actually touches it.
