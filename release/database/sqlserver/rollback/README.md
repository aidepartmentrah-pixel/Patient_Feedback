# Rollback

`rollback_install.sql` reverses a **fresh install** that hasn't gone into real use yet — it
drops every table `install/002_create_schema.sql` creates. This is for aborting a failed or
test install (e.g. the scratch-database dry-run mentioned during package development), not
for undoing changes on a database with real hospital data in it.

**Do not run this against an installation with real data.** If a real installation needs to
be rolled back, that is a restore-from-backup operation — see
`../scripts/restore_database.sql` — not a schema rollback.

For rolling back a specific future *migration* (as opposed to the whole install), add a
matching `rollback_<migration_name>.sql` alongside it in this folder when that migration is
written, per `../migrations/README.md`.
