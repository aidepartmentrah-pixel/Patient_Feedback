# Doctor & Worker Search — Questions for the Development Team

**Prepared:** 2026-07-23
**Purpose:** Doctor and Worker search are currently broken (confirmed live). The fix is understood in outline — mirror the dual-source pattern already built for patients (reserve table + Hospital Directory API, external failure never hides reserve results) — but several specifics need to come from the team before implementation.

---

## 1. Current confirmed state (for context)

- **Doctor search crashes outright**: the query references `VW_Doctors`, which does not exist in this database at all (`Invalid object name 'VW_Doctors'`). `APP_RESERVE_DOCTOR` exists but has 0 rows. `APP_LOOKUP_DOCTOR` exists (columns: `DoctorID`, `DoctorName`, `Specialty`, `IsActive`, `SourceSystem`, `LastSyncedAt`) but also has 0 rows — it looks like a sync-cache table that was scaffolded but never wired up to anything that populates it.
- **Worker/employee search silently returns empty, no error**: the query references `VW_HrEmployeeProfileView`, which does exist but has 0 rows. There is no reserve table equivalent referenced in the current search code at all.
- The Hospital Directory API's own spec (`Hospital_Directory_API_OpenAPI.yaml`) defines `/doctors` and `/workers` search endpoints with `Doctor` and `Worker` schemas, but `hospital_directory_client.py`'s own docstring says these were never implemented: *"Doctor/worker resource calls are NOT implemented here yet — those are separate, later sessions (C2/C3)."*
- The user has confirmed the intended architecture directly: **both doctors and workers are dual-source** — a writable local reserve table (for entries added directly in this app) merged with live Hospital Directory API calls — exactly the pattern already built and working for patients in `patient_directory_service.py`.

## 2. What we need from the team

### Reserve tables
1. Does a reserve table for **workers/employees** already exist under a different name (analogous to `APP_RESERVE_DOCTOR` / `APP_RESERVE_PATIENT`), or does it need to be created from scratch?
2. For `APP_RESERVE_DOCTOR`, please confirm the authoritative current column list/types (we see it referenced with `DoctorID`, `DoctorName`, `Specialty`, `IsActive` in existing code — is this still accurate?).
3. Is `APP_LOOKUP_DOCTOR` (the 0-row sync-cache table with `SourceSystem`/`LastSyncedAt` columns) meant to be retired/removed now that the plan is live API calls instead of a synced cache, or does it serve a different purpose we should preserve?

### Hospital Directory API integration
4. Confirm the `/doctors` and `/workers` endpoints in `Hospital_Directory_API_OpenAPI.yaml` are current and authoritative (same status as the `/patients` endpoint we already integrated against).
5. Do doctor/worker external identities need the same opaque-ID encoding pattern used for patients (`ext__{id}`, see `hospital_directory_client.encode_external_patient_id`), or can `doctor_id`/`employee_id` from the API be used directly as plain strings? This depends on whether any table stores these as an integer foreign key anywhere downstream (worth double-checking `APP_IncidentCaseEmployee` and equivalent doctor-linkage tables).
6. Specialty/job-title data: the `Doctor` schema has `specialty_id`/`specialty_name`, `department_id`/`department_name`; the `Worker` schema has `job_id`/`job_title`, `department_id`/`section_id`/`administration_id`. Do these need to map to any local lookup tables, or are they just display strings passed through as-is (matching how patient search doesn't map `sex` to anything beyond a display label)?

### Quick-add / manual entry
7. Patient search has a "quick add" flow for creating a new reserve patient when the external API doesn't have someone. Should doctor/worker search get the same treatment, or is quick-add out of scope for this fix?

### Anything else
8. Any other constraints, in-progress work, or existing partial implementation for doctor/worker external search we should know about before starting (e.g., is there a C2/C3 branch or design doc already underway)?

---

Once we have answers here, the implementation itself is a known quantity — a `doctor_directory_service.py` / equivalent, structured exactly like `patient_directory_service.py`, with `search_service.py`'s `search_doctors`/`search_employees` delegating to it instead of querying the dead views directly.
