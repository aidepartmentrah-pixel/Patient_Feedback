# SQL Server Database Package — Patient Feedback / IncidentManager

Overview document for `database/sqlserver/`, the SQL Server Database Package prepared as Step 1A
of the two-step Dockerization pipeline (RAH Lab Application Dockerization & Deployment Playbook).
Step 2A (Dockerize) consumes this package; it does not Dockerize anything itself.

## What this package is for

Preparing this project for deployment on the Lenovo Legion (Docker Desktop) and eventually the
hospital's offline infrastructure, while this VM (Windows Server, no Docker support) cannot be
the build machine. This package makes the database side of that move reproducible: given a blank
SQL Server instance, `install/` produces a working schema with the reference/configuration data
the app needs to function — without ever carrying real hospital patient/complaint data off this
box, per the RAH Infrastructure Standard's production-data-protection rules.

## Directory map

```
database/
├── sqlserver/
│   ├── install/        Fresh-install schema + seed data (001-011, run in order)
│   ├── migrations/      Future schema-evolution scripts (extends dbo.SchemaMigrationHistory)
│   ├── retirement/      Destructive obsolete-object removal — reviewed, never auto-run
│   ├── validation/      Post-install structural checks
│   ├── rollback/        Undo a failed/test fresh install (not for real installations)
│   └── scripts/         backup_database.sql / restore_database.sql / verify_database.sql
└── docs/                This file + the guides below
```

## The three mechanisms, and why they're kept separate

| Mechanism | Moves | When to use |
|---|---|---|
| `install/` | Schema + universal lookup/config seed data only | Every fresh installation |
| `migrations/` + `retirement/` | Schema changes to an existing installation | Evolving a live installation without losing its data |
| `scripts/backup_database.sql` + `restore_database.sql` | An installation's actual real data (incidents, ML training history, patient reserve records, users) | Moving *this specific hospital's* installation to another server |

Mixing these up is exactly what this package is designed to prevent — see
`DATABASE_STRUCTURE_REPORT.md` for the full per-table classification that enforces the boundary,
and `DATABASE_UPDATE_GUIDE.md` for the decision tree.

## Database summary

- SQL Server Express, local instance, database `IncidentManager`.
- 83 tables total; 81 travel with a fresh install (2 confirmed obsolete — `VW_PatientAdmission`,
  `VW_Doctors` — excluded from `install/`, handled via `retirement/`).
- Zero views, stored procedures, functions, or triggers — all business logic is in the Python
  backend (`backend/api/db_layer/`, `backend/api/services/`).
- A migration-tracking table (`dbo.SchemaMigrationHistory`) already existed in production before
  this package was built; this package extends that convention rather than replacing it.

See `DATABASE_STRUCTURE_REPORT.md` for the full table-by-table inventory and classification.
