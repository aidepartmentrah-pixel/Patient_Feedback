# Retirement Migrations

Scripts here remove objects confirmed obsolete in the live application, but they are
**destructive against whatever database they run on** and are never executed automatically
by the install or migration pipeline.

## Rule

A retirement script may be written and reviewed at any time. It may only be **executed**
against a real installation after:

1. Re-running its own dependency-check query and confirming zero blocking rows.
2. Taking a full database backup (`../scripts/backup_database.sql`).
3. Explicit sign-off from whoever owns that installation.

## Current contents

`001_retire_vw_patientadmission_and_vw_doctors.sql` — despite the filename (kept to avoid
churn — rename it once a proper VW_Doctors script exists), this now removes **only**
`dbo.VW_PatientAdmission`, confirmed obsolete 2026-07-21:

- `VW_PatientAdmission`: patient reads moved to the Hospital Directory API in an earlier
  session; `backend/api/db_layer/patients_db.py` carries an explicit comment that it no
  longer imports `PATIENT_ADMISSION_TABLE` at all.
- Verified 2026-07-21: zero foreign keys reference this table (`sys.foreign_keys` query
  embedded in the script itself, re-run it before executing).

**`VW_Doctors` was removed from this script's scope on 2026-07-23.** Its original
justification here ("doctor reads now go through `APP_LOOKUP_DOCTOR`... `VW_Doctors` only
appears in stale code comments") did not match the actual code —
`backend/api/services/search_service.py`'s `search_doctors()`, the function behind the
incident-creation form's doctor autocomplete, still ran a live `SELECT ... FROM VW_Doctors`
query. That's the exact bug that broke doctor search on a fresh install with no
`VW_Doctors` data. `search_doctors()` has now been migrated to the Hospital Directory API
(Session C2 — see `backend/api/services/staff_directory_service.py`), same commit as this
correction. Once that's verified live on a real installation, `VW_Doctors` becomes a
legitimate retirement candidate again — write a **new**, freshly dependency-checked script
for it rather than re-adding it here from stale assumptions.

The `DROP TABLE` statement in that file is commented out on purpose. Uncommenting and
running it is a separate, explicitly-approved step — not part of this package's generation.
