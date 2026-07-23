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

`001_retire_vw_patientadmission_and_vw_doctors.sql` — removes `dbo.VW_PatientAdmission` and
`dbo.VW_Doctors`, confirmed obsolete 2026-07-21:

- `VW_PatientAdmission`: patient reads moved to the Hospital Directory API in an earlier
  session; `backend/api/db_layer/patients_db.py` carries an explicit comment that it no
  longer imports `PATIENT_ADMISSION_TABLE` at all.
- `VW_Doctors`: doctor reads now go through `APP_LOOKUP_DOCTOR` (a sync cache populated
  from the external API, distinct table). `VW_Doctors` only appears in stale code comments —
  grep for `FROM.*VW_Doctors` across `backend/api/` returns nothing.
- Verified 2026-07-21: zero foreign keys reference either table (`sys.foreign_keys` query
  embedded in the script itself, re-run it before executing).

The `DROP TABLE` statements in that file are commented out on purpose. Uncommenting and
running them is a separate, explicitly-approved step — not part of this package's generation.
