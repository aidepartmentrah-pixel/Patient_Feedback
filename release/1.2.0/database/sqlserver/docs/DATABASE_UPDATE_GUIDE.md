# Database Update Guide

Decision tree for "I need to change something about a database" — pick the right mechanism,
since mixing them up either loses real data or ships patient data somewhere it shouldn't go.

## Which situation are you in?

**"I need a brand-new, empty installation."**
→ `DATABASE_INSTALL_GUIDE.md`. Schema + lookup/config seed only.

**"I need to add/change a table or column on a database that already has real data,
and keep that data."**
→ Write a new file in `sqlserver/migrations/`, following `migrations/README.md`. Wrap DDL in
existence checks, end with an insert into `dbo.SchemaMigrationHistory` under the same name as
the file. Never `DROP` anything that might hold real data this way — see the next case.

**"I need to remove an object that's now obsolete (dead code no longer references it)."**
→ `sqlserver/retirement/`, following `retirement/README.md`. Write it, review it, confirm the
dependency check returns zero rows, take a backup, then get sign-off before running the
(commented-out) `DROP` statements. Never bundle a `DROP` into a regular migration.

**"I need to move an existing hospital's whole installation to a different server."**
→ `DATABASE_BACKUP_GUIDE.md` + `DATABASE_RESTORE_GUIDE.md`. Full `BACKUP DATABASE`/
`RESTORE DATABASE` — this is the only path that legitimately carries real patient/complaint
data and ML training history, because it stays within that one installation's own data, it
isn't being distributed as a generic release artifact.

**"I need to prepare a release/install package that goes to a new environment (Legion, offline
validation, eventual hospital deployment)."**
→ `DATABASE_INSTALL_GUIDE.md`'s `install/` scripts, never the backup/restore path. A release
package must never contain real hospital data — see `DATABASE_STRUCTURE_REPORT.md`'s
classification manifest for what is and isn't safe to include.

## The rule this all traces back to

> Schema migrations create and alter structures. Seed scripts insert universal lookup/
> configuration/reference data. Upgrade or restoration processes preserve the data already
> belonging to a particular hospital installation. Backup and restore move the complete hospital
> database when transferring that installation to another server. Release packages must not
> contain real hospital complaint data.

This mirrors why the 961-row `ml.HistoricalTrainingExample` and 26-row `ml.CaseTrainingRecord`
tables exist in the first place: they were moved from a retired SQLite ML store into SQL Server
as part of an *existing-installation* upgrade migration (to preserve this installation's ML
asset), not seeded as universal reference data — they must never travel with a generic install
package.
