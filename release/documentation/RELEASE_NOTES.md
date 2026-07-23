# Patient Feedback System — Release 1.0.0

**Release date:** 2026-07-22
**Prepared by:** RAH Lab
**Application Release:** 1.0.0 (baseline — first Dockerized release)
**Infrastructure Release compatibility:** RAH-OIP 1.0.0 or later

## What this release is

The first Dockerized, offline-installable release of the Patient Feedback
System (previously deployed on Windows Server via IIS + NSSM). This release
converts that application into three Docker images plus a database
installer, deployable through Docker Compose on a Debian offline server.

## Contents

| Component | Version | Notes |
|---|---|---|
| Backend | 1.0.0 | `rah-pfms-backend:1.0.0` |
| Frontend | 1.0.0 | `rah-pfms-frontend:1.0.0` |
| DB init/migrate | 1.0.0 | `rah-pfms-db-init:1.0.0` |
| Database schema | baseline_install_1.0.0 | see `database/sqlserver/` |

## What's new

- Full Docker Compose deployment: SQL Server + automated database
  installer + backend + frontend, with health checks and persistent
  volumes for database data and backend configuration.
- Backend and frontend both run as non-root container users.
- Frontend's Cairo font (previously loaded live from `fonts.googleapis.com`)
  is now self-hosted, per the org's Offline Frontend Asset Standard.
- Nginx reverse-proxies `/api/*` from the frontend container to the backend
  container, replacing the old IIS URL Rewrite rule.
- Speech-to-Text (Faster-Whisper `medium`) runs fully offline: the model is
  pre-exported to `assets/whisper-model-medium.zip` (via
  `scripts/export_whisper_model.sh`, run once on an online engineering
  machine) and extracted locally by `install_offline.sh` — the backend loads
  it from disk via `WHISPER_MODEL_PATH` and never contacts huggingface.co.
  Same pattern already proven in the voice-project deployment.
- **Real organizational units and user accounts are now provisioned as a
  mandatory part of installation** — migrated from the old HCAT system
  (`170.70.32.34`) via a three-stage pipeline (one-time online extraction →
  validated `provisioning.v1.json` artifact → transactional provisioning at
  install time). 179 organizational units, 162 user accounts, and 161
  role/scope assignments are provisioned automatically by `db-init`; a fresh
  install can no longer complete with zero usable accounts. See
  `database/sqlserver/seed/` for the full pipeline and
  `provisioning.v1.manifest.json` for exact record counts. All passwords are
  real bcrypt hashes (never `TEMP_HASH_`, never plaintext) — see "Password
  handling" below.
- Fixed: the Hospital Directory API settings page (Config → Hospital
  Directory API) previously 500'd on every request because
  `APP_ExternalApiSettings` had no seed row — fixed via
  `install/013_create_external_api_settings_table.sql`. Saving a Hospital
  Directory API key additionally requires `SETTINGS_ENCRYPTION_KEY` (see
  `.env.offline.template`) — without it, the save endpoint fails clearly
  rather than silently.
- **Real Custom Table Views are now provisioned as part of installation** —
  the 11 real, currently-active views from the old HCAT system (Classifications,
  Decision & Root Cause, Escalation Alerts, Administration Review, Department
  Review, Section Review, Case Timeline, Clinical Risk, Satisfaction Follow-Up,
  Recently Edited, Operational Overview) are extracted, validated, and
  provisioned the same way org units/users are — see
  `install/015_create_custom_view_source_mapping_table.sql` and
  `seed/extract_custom_views.py`. ~21 deactivated test/experimental views on
  the source system (test data like duplicated "Abbass" entries, gibberish
  text) are deliberately excluded — only active views are migrated.
- Fixed: Dashboard Scope selector (Administration/Department/Section
  cascading dropdowns) came back empty against real migrated org-unit data —
  `dashboard_service.py` detected the Administration root via a self-parented
  `ParentID == UniqueID` convention that real data doesn't follow (real roots
  have `ParentID = NULL`). Now detects by `Type == 323` instead.
- Fixed: Patient History search and patient selection during complaint entry
  crashed (`KeyError`) after the Hospital Directory API's contract changed —
  the API dropped its patient+visit identity model in favor of a plain
  `patient_id`. `hospital_directory_client.py` and
  `patient_directory_service.py` updated to match the current contract.
- Fixed: Drawer Notes had zero usable labels on a fresh install (feature has
  no equivalent on the old HCAT system, so nothing to migrate) — seeded four
  reasonable defaults via `install/014_seed_default_drawer_labels.sql`
  ("Follow-up Required", "Resolved", "Escalated", "Internal Note").
- Fixed: Model Dashboard rendered a literal, alarming "0/0/0%" on a fresh
  install instead of an honest empty state — the backend response was
  already correct (`status: "never_run"`), the frontend now renders a clear
  "No training runs yet" message instead of raw zeros.
- **Fixed: ML Classification (category/subcategory) — root design flaw
  corrected, zero patient data in the deployable artifact.** The pretrained
  hierarchical category/subcategory models translate a raw model output
  (0, 1, 2, ...) into a real label via a small `temp_to_label` mapping
  (e.g. `{0: 12, 1: 15, 2: 22}`). That mapping is fully determined at
  training time — it never changes after training — but the code
  previously reconstructed it by querying `table_feedback_train` live, out
  of a 116MB SQLite file of real patient complaint text, on every server
  startup. That file was never packaged into this deployment (too large for
  git, absent from the build machine), which crashed all 9 classification
  outputs together (not just the 2 that needed it).
  **Fix**: each of the 10 category/subcategory models now ships with a tiny
  JSON sidecar (e.g. `vocab_models/category_domain1_label_map.json`)
  generated once at training time — `label_mapping_helper.py` reads that
  JSON instead of querying SQLite. No patient data of any kind travels with
  this release for classification to work. Verified end-to-end with the
  116MB source file completely absent from the container.
  8 of the 10 models' label maps were recovered and verified against the
  live model files (`model.n_classes_` matches exactly): all 3 domain→category
  models, and 5 of 7 category→subcategory models (categories 3,4,5,6,7).
  **2 of 10 remain genuinely stale** — Category 1's and Category 2's
  subcategory models expect a different class count than any available
  data supports (confirmed independently on the live production system,
  same failure). These two return `"Not Available"` honestly rather than a
  wrong guess, and need real retraining against current data — see
  `ML_CLASSIFICATION_ISSUE_FOR_DEV_TEAM.md` for the full technical
  handoff. Net result: 8 of 9 classification outputs work with real
  predictions (domain, category for all domains, subcategory for 5 of 7
  categories, severity, harm level, stage, feedback type, improvement
  opportunity, classification-EN); subcategory for categories 1/2
  specifically remains unavailable pending model retraining.
- Fixed: Speech-to-Text (audio recording → text) stopped inserting any
  transcribed text. Two real frontend bugs, both in how recorded audio was
  packaged before upload: (1) `TextBlocksWithButtons.js` recorded with the
  browser's default codec (webm/opus on Chrome/Edge, ogg/opus on Firefox)
  but mislabeled the result as `audio/wav`/`recording.wav` — the bytes never
  matched the claimed format; (2) `useSttRecorder.js` (used by the current
  Insert Record page) had a bare `catch { /* ignore */ }` around the
  transcription call — any failure (network, session, backend error)
  vanished silently with no error shown and no transcribed text inserted,
  which matches the reported symptom exactly. Both now record with a
  browser-verified `mimeType`, label the uploaded file to match the real
  encoding, and surface real errors instead of swallowing them. Verified
  against the actual backend endpoint with real encoded audio (not just
  code review) — confirmed working end-to-end.
- **Verification scripts extended to actually check every fix in this
  release**, not just infrastructure health. `verify_installation.sh` now
  checks Custom Table View counts and Drawer Note label presence against
  the database directly. `qualify_offline_installation.sh` now runs real
  functional checks through the actual API: Dashboard Scope returns real
  cascading data, Custom Views and Drawer Labels are reachable via their
  real endpoints, ML Classification returns real predictions without
  crashing, Speech-to-Text actually transcribes a test audio file, NER
  endpoints are confirmed gone (404, not just hidden), Publication Batches
  and Patient Search respond correctly for a logged-in user. All of these
  were previously only checked manually during this release's development.
- **NER (Named Entity Recognition) removed completely** — was an optional,
  never-fully-wired auto-fill layer on top of the patient/doctor/staff
  search boxes (which remain, manual-search-only). Removed: `ner_router.py`,
  `ner_service.py`, `models_directory/NER_Model/` entirely, the `gliner`/
  `onnxruntime`/`stanza` dependencies from `backend/requirements.txt`, and
  all `extractNER`/NER UI wiring from the frontend (Insert Record, Migration
  Form). Reduces the backend image's dependency footprint.

## Password handling

All 162 migrated accounts use real bcrypt hashes, generated during the
Stage A→B transform from the old system's recoverable test passwords — none
use the `TEMP_HASH_` scheme in this deployment. The provisioning artifact
(`provisioning.v1.json`) never contains plaintext passwords. A small,
separately-protected `installation_test_credentials.local.json` (one active
account per role, plaintext) exists **only** for
`qualify_offline_installation.sh`'s post-install login check — it is not
read by the normal installer and should be deleted or handed back securely
after qualification testing, same handling as a temporary credential handoff.

## Known Gaps (read before offline deployment)

1. **13 flagged records from the source-system migration, preserved as-is,
   not fabricated or silently cleaned up** — see
   `provisioning.v1.json`'s `flagged_records` array for the full list.
   Notably: 6 organizational units (`22, 138, 167, 168, 172, 177`) exist in
   the source but are structurally incomplete (NULL/missing `Type` and
   `ParentID`) — not frozen/archived, genuinely incomplete data in the old
   system; `sec_289_admin` is an **active** account whose organizational
   scope (`OrgUnitID=289`) does not exist anywhere in the source system at
   all — preserved faithfully rather than fabricating a fix; 4 accounts have
   test-looking usernames/org units (`universal_section_user`,
   `sec_310/311/312_admin`) and are already inactive in the source.
2. **This is a baseline release** — `database/sqlserver/migrations/` is a
   placeholder (see its `README.md`). `update_offline.sh` re-runs the
   install scripts (safe, they're idempotent) rather than applying
   incremental migrations. The next release that changes the schema must
   add real migration scripts and update `update_offline.sh` accordingly.
3. **SQL Server edition/licensing is not decided by this release.** `.env`
   defaults to `Developer` locally and the offline template defaults to
   `Express`; confirm the hospital's actual SQL Server licensing before
   production deployment (see `.env.offline.template`).
4. **Whether this app gets its own SQL Server container or shares OR-LAB's
   existing HCopilot SQL Server instance is not yet decided** — see the
   comment in `.env.offline.template`. RAH's Database Standard prefers
   sharing one instance across independent databases when practical.
5. **ML Classification's category/subcategory outputs are not restorable in
   this release** — see "What's new" above. 7 of 9 classification outputs
   work; category and subcategory return `"Not Available"`. This is a
   pre-existing stale-model problem (confirmed present on the live
   production system too), not something introduced by Dockerization.

## Release Package Size

Total release package: **~4.6GB** (`backend.tar` 3.2GB, `whisper-model-medium.zip`
1.4GB, `db-init.tar` 58MB, `frontend.tar` 24MB — the SQL Server image is
intentionally excluded, see `INSTALL_OFFLINE.md`).

This is uncomfortably close to a single-layer DVD's 4.7GB capacity (little
to no margin once filesystem/ISO overhead is accounted for) — **use
dual-layer DVD (8.5GB), USB, or another approved medium with real headroom**,
not single-layer DVD, for this release. Most of the size is `backend.tar`'s
full ML dependency stack (torch, transformers, xgboost, scikit-learn, scipy,
pandas, matplotlib, streamlit, ctranslate2, etc.) — `gliner`, `onnxruntime`,
and `stanza` were removed as part of this release's NER removal, trimming
the stack somewhat; further reduction would mean pruning additional unused
dependencies from `backend/requirements.txt`, out of scope for this release.

## Ports (reserved on OR-LAB per the port snapshot doc)

| Service | Port |
|---|---|
| Backend | 8100 |
| Frontend | 8101 |
| (spare) | 8102 |

Re-verify these are still free on OR-LAB itself before deploying — the
source port snapshot has no date and is not a live guarantee.
