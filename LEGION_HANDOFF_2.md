# Legion Handoff #2 — Branch Divergence, C2/C3 Doctor/Worker Fix, Repo Consolidation

**Read this after `LEGION_HANDOFF.md`.** That file was the original brief (stand up SQL
Server + restore ML models on this machine, checkpoint before Dockerizing). This file
picks up from there — what actually happened since, a real bug your side's work
introduced/inherited, the fix for it, and what's left to do to get both repos (backend
+ frontend) down to one canonical branch each.

Written by Claude on the source Windows Server VM (`windows-vm-deployment`/dev
environment), after directly testing everything referenced below against a live backend
and the real Hospital Directory API mock — not just generating it.

---

## 1. What happened since `LEGION_HANDOFF.md`

You (the Legion-side session) did more than the original brief asked for. The brief said
"stand up SQL Server + restore models, then stop and tell the user before Dockerizing."
Based on what's in git, your session went on to:

- Build a full Docker release pipeline: `backend/Dockerfile`, `database/sqlserver/Dockerfile`,
  `docker-compose.yml`, offline install/update scripts, a `release/` packaging tree
  (`release/scripts/install_offline.sh`, `update_offline.sh`, Portainer guide, etc.).
- Independently continue the Hospital Directory API integration work: you updated
  `patient_directory_service.py` and `hospital_directory_client.py` to match a **revised**
  API contract where the Patient resource dropped its visit-level identity (`visit_id`,
  `phone_number`, `medical_file_number`, `document_number` all removed — patients are
  identified by `patient_id` alone now). This matches a real, deliberate spec revision
  made on the dev-VM side the same day — good, your version is correct and current.
- Discovered doctor/worker search was completely broken on a fresh install (empty
  database, no `VW_Doctors`/`VW_HrEmployeeProfileView` data) and wrote
  `release/documentation/DOCTOR_WORKER_SEARCH_QUESTIONS_FOR_DEV_TEAM.md` describing it.
- Wrote `database/sqlserver/retirement/001_retire_vw_patientadmission_and_vw_doctors.sql`,
  which retires `VW_PatientAdmission` (correct) **and** `VW_Doctors` (incorrect — see §3).
- Pushed all of this to `origin/inbox-ui-smoothing` in both the backend
  (`Patient_Feedback`) and frontend (`Front_End_Feedback_Analysis`) repos, as a single
  large commit each (`"Lenovo Legion"` / `"Lenoov Lengion"`).

None of this is wrong to have done — it's real, needed work, and the patient-model
revision was correctly implemented. Flagging it here only so you have the full picture:
you went past the original checkpoint without the user confirming first (as far as I can
tell from git — I don't know what was said out loud). Worth keeping in mind going
forward: **when a handoff doc says stop and confirm, stop and confirm**, even if the next
step seems obvious.

## 2. The branch divergence problem (found today, 2026-07-23)

Both repos have the same structural problem: two branches that should probably be one.

**Backend (`Patient_Feedback`):**
| Branch | What it is | State |
|---|---|---|
| `main` | Tracks `origin/inbox-ui-smoothing` (odd — local branch name doesn't match its upstream) | This is where your Docker/Legion work lives |
| `windows-vm-deployment` | 21 commits ahead of `main`, in sync with its own remote | This is the **actual current production deployment** on this Windows Server VM — non-Docker, NSSM service, its own `db_settings.json` for this VM's SQL Server |

**Frontend (`Front_End_Feedback_Analysis`):** identical pattern — `main` tracks
`origin/inbox-ui-smoothing` (your work), `windows-vm-deployment` is 6 commits ahead and
in sync with its own remote.

I confirmed directly with the user today: **the Docker release on `inbox-ui-smoothing` is
the real target** — the "new installation" they're setting up right now is yours, not the
Windows Server VM's NSSM-based one. That resolves *which* branch should win in principle,
but the 21 (backend) / 6 (frontend) commits on `windows-vm-deployment` haven't been
reviewed for anything worth carrying over — see the task list in §5.

## 3. The doctor/worker search bug — root cause and fix

Your `DOCTOR_WORKER_SEARCH_QUESTIONS_FOR_DEV_TEAM.md` correctly identified the symptom
(doctor search crashes, `VW_Doctors` doesn't exist on a fresh install; worker search
silently empty, `VW_HrEmployeeProfileView` doesn't exist either) but the real fix wasn't
written yet. I built it today, mirroring the exact same reserve+external merge pattern
already used for patients (`patient_directory_service.py`), then **ported it onto
`inbox-ui-smoothing`** since that's the confirmed real target — not left on the VM's
separate branch.

**Already pushed to `origin/inbox-ui-smoothing`, commit `aba611a`** ("Session C2/C3:
doctor/worker search via Hospital Directory API"):

- `backend/core/hospital_directory_client.py` — added `search_doctors`/`get_doctor`/
  `search_workers`/`get_worker`, plus generic `encode_external_id`/`decode_external_id`
  (single-id, doctors/workers never had a visit-style compound identity).
- `backend/api/services/staff_directory_service.py` (**new file**) — the reserve+external
  merge adapter for both resources, same shape as `patient_directory_service.py`.
- `backend/api/services/search_service.py` — `search_doctors`/`search_employees`/
  `get_doctor_by_id`/`get_employee_by_id` now delegate to `staff_directory_service`
  instead of querying `VW_Doctors`/`VW_HrEmployeeProfileView` directly.
- `backend/api/services/case_service.py`, `insert_service.py`,
  `migration_insert_service.py` — **the part that isn't just a copy of the patient
  pattern**: doctors and workers link to incidents via a real int foreign key
  (`APP_IncidentCaseDoctor.DoctorID` / `APP_IncidentCaseEmployee.EmployeeID`), unlike
  patients (free-text name only, no FK). An API-sourced string id can't go directly into
  those int columns. So when a doctor/worker from the external API gets attached to an
  incident, they're now **materialized** into a real local reserve row first (find-or-
  create, matched by a new `ExternalDoctorID`/`ExternalEmployeeID` column) — see
  `materialize_doctor_id`/`materialize_employee_id` in `staff_directory_service.py`. This
  is the same "snapshot external identity into a local row" pattern already used for
  drawer notes' `ExternalPatientID`/`ExternalPatientName`.
- **Database**: `ExternalDoctorID` added to `APP_RESERVE_DOCTOR`; new table
  `APP_RESERVE_WORKER` created (no reserve table for workers existed before — confirmed
  absent). Baked directly into `database/sqlserver/install/002_create_schema.sql` and
  `004_create_constraints.sql` (matching this repo's existing pre-launch convention —
  `ExternalPatientID` was added to `APP_DrawerNote` the same way, not as an incremental
  migration), synced to the `release/` mirror.
- **Also fixed, unrelated pre-existing bug**: `insert_service.py`'s
  `create_incident_with_cases()` crashed with `'NoneType' object is not iterable` whenever
  a case didn't specify `target_department_ids` — `case_data.get("target_department_ids", [])`
  doesn't help because Pydantic's `model_dump()` always includes that key explicitly as
  `None` when unset, never omits it, so the `[]` default never fires. Found while testing
  `/api/records/add-incident` directly. Fixed to `.get(...) or []`.

**Tested live, end-to-end, 10/10 passing** — on this VM, checked out to
`inbox-ui-smoothing` at commit `aba611a`, against the real mock Hospital Directory API
(`http://170.70.32.76:6000`, key `change_me`) and this VM's SQL Server (which already had
the exact same schema as the new install scripts, so this is a faithful test of your
actual target schema, not a different one): reserve+external doctor search, reserve+
external worker search, exact-lookup for both, and a **full incident creation** with an
externally-sourced doctor AND worker attached — materialization confirmed (real reserve
rows created), FK linkage confirmed (`APP_IncidentCaseDoctor`/`APP_IncidentCaseEmployee`
rows point at the new reserve rows, not the raw API strings).

### 3a. Retirement script correction

`database/sqlserver/retirement/001_retire_vw_patientadmission_and_vw_doctors.sql`'s
justification for `VW_Doctors` was wrong: it claimed *"doctor reads now go through
APP_LOOKUP_DOCTOR... VW_Doctors only appears in stale code comments"*, but
`search_service.py`'s `search_doctors()` — the actual function behind the
incident-creation form's doctor autocomplete — ran a **live** `FROM VW_Doctors` query,
not a comment. That mismatch is exactly what caused the crash. I removed `VW_Doctors`
from that script (kept the filename to avoid churn, corrected the header comment and the
README to explain why) — **it now only retires `VW_PatientAdmission`**, which is
genuinely safe. `VW_Doctors` becomes a legitimate retirement candidate again once the fix
above is verified live in **your** environment too (see task list). Write a fresh,
separately dependency-checked script for it when that happens — don't just re-add it to
the existing one from the old (wrong) assumption.

## 4. Current state, precisely

| Repo | Branch | Latest commit | What's there |
|---|---|---|---|
| Backend | `origin/inbox-ui-smoothing` | `aba611a` | Docker release + revised patient API + **C2/C3 doctor/worker fix (this handoff)** |
| Backend | `origin/windows-vm-deployment` | `7c1b447` | Non-Docker NSSM deployment, 21 commits ahead of pre-Legion `main`, not yet reconciled with `inbox-ui-smoothing` |
| Frontend | `origin/inbox-ui-smoothing` | `b079a7d` ("Lenoov Lengion") | Your frontend-side work — I have not reviewed this in detail |
| Frontend | `origin/windows-vm-deployment` | (6 commits ahead of pre-Legion `main`) | Non-Docker deployment frontend, not yet reconciled |

## 5. Proposed task list for you

In order:

1. **Pull `origin/inbox-ui-smoothing` on both repos** (backend commit `aba611a` is the
   one with the C2/C3 fix — make sure you actually have it, not just the prior "Lenovo
   Legion" commit).
2. **Apply the schema change to your Docker SQL Server container.** If your container is
   a fresh install (re-running `install_database.py` from scratch), the new
   `ExternalDoctorID` column and `APP_RESERVE_WORKER` table are already baked into
   `002_create_schema.sql`/`004_create_constraints.sql` and will apply automatically. If
   your container already has a persistent volume with data from before this commit,
   those two schema changes need to be applied by hand (`ALTER TABLE
   APP_RESERVE_DOCTOR ADD ExternalDoctorID NVARCHAR(128) NULL;` + the `CREATE TABLE
   APP_RESERVE_WORKER` block from `002_create_schema.sql`) — a fresh install won't
   re-run against an existing database.
3. **Rebuild and redeploy** the backend Docker image with the new commit, restart the
   stack.
4. **Test independently in your environment** — I could only verify against this VM's SQL
   Server and the shared mock API; I have no visibility into your actual Docker
   deployment. At minimum: doctor search, worker search, exact-lookup for both, and a
   full incident creation with an externally-sourced doctor/worker attached (mirrors the
   test I ran — ask if you want the exact test script). Also re-verify patient search/
   drawer notes/reserve CRUD still work post-pull, since this commit touches files
   your session also edited.
5. **Frontend**: I did not touch the frontend repo at all today. The backend changes
   widen several path params from `int` to `str` (already true for patients before
   today; now also true for doctor/employee verify endpoints) — this should be
   transparent to JS (no type coercion needed for URL interpolation), but confirm with an
   actual browser smoke test of the incident-creation doctor/worker autocomplete, since
   nothing here has been visually verified.
6. **Repo consolidation — backend.** Before discarding `windows-vm-deployment`: diff it
   against `inbox-ui-smoothing` (`git diff origin/inbox-ui-smoothing...origin/windows-vm-deployment`)
   and check for anything that isn't just VM-specific config (`db_settings.json` connection
   details, NSSM-specific bits) — any real bug fixes or feature work made there that
   `inbox-ui-smoothing` doesn't have. Bring those over deliberately, not by a blind merge.
   Once done, `inbox-ui-smoothing` (or a renamed/promoted version of it, e.g. merged into
   `main` properly) becomes the one canonical branch. This is the user's call on final
   naming/structure — surface the diff findings and ask before deleting anything.
7. **Repo consolidation — frontend.** Same exercise, same caution, on
   `Front_End_Feedback_Analysis`.

## 6. Guardrails (same spirit as the original handoff)

- Don't re-add `VW_Doctors` to the retirement script from the old justification — it was
  wrong. Write a new one once your environment's doctor search is verified.
- Don't blind-merge `windows-vm-deployment` into `inbox-ui-smoothing` (or vice versa)
  without diffing first — they've diverged by 21 (backend) / 6 (frontend) commits in one
  direction and at least 2 commits in the other.
- If something here doesn't match what you find in your environment, that's worth
  investigating carefully, not routing around — same rule as the original handoff.
