# Validation Checklist — Patient Feedback System

Complete this checklist after every install and every update, before
declaring the deployment done. Check off each item; do not skip any.

## Engineering / Release Preparation

- [ ] `APP_VERSION` in `.env` matches the version tag on the loaded images.
- [ ] `MSSQL_SA_PASSWORD` is set to a strong, unique value (not the example
      placeholder).
- [ ] `MSSQL_PID` matches the hospital's actual SQL Server license.
- [ ] `SETTINGS_ENCRYPTION_KEY` is set to a real generated Fernet key (not
      the `__SET_ME__` placeholder).
- [ ] Release Register updated with this deployment (project, versions,
      date, engineer, rollback availability — see RAH Lab Operator Manual
      §6.4).

## Offline Deployment

- [ ] `docker --version` and `docker compose version` both succeed.
- [ ] SQL Server image (`mcr.microsoft.com/mssql/server:2022-latest`)
      already present on the server (from the Offline Debian Server Kit).
- [ ] `./scripts/load_images.sh` completed with no errors.
- [ ] `./scripts/install_offline.sh` (first install) or
      `./scripts/update_offline.sh` (update) completed successfully.
- [ ] A pre-change database backup exists (automatic on update; run
      `./scripts/backup_database.sh` manually before any first install too,
      once there is real data to protect).

## Production Validation

- [ ] `docker compose ps` shows `sqlserver`, `backend`, `frontend` all
      healthy, and `db-init` exited with code 0.
- [ ] `./scripts/verify_installation.sh` reports `0 failed`.
- [ ] `curl http://localhost:8100/api/status` returns
      `"connected":true` and `"bootstrap_mode":false`.
- [ ] `curl http://localhost:8101/` returns the frontend HTML.
- [ ] `curl http://localhost:8101/api/status` returns the same JSON as the
      direct backend check (confirms the reverse proxy works).
- [ ] Application loads in a browser at `http://<server-ip>:8101` and the
      login page renders correctly, including the Cairo font (Arabic and
      Latin text should render in the intended typeface, not a fallback
      font — if it looks like a generic system font, the self-hosted font
      files may not have made it into this build).
- [ ] Login works with a known test account for **each distinct role**
      (WORKER, COMPLAINT_SUPERVISOR, SECTION_ADMIN, DEPARTMENT_ADMIN,
      ADMINISTRATION_ADMIN) — see `qualify_offline_installation.sh`.
- [ ] Role-scoped visibility actually differs per role (e.g. a
      SECTION_ADMIN's `allowed_unit_ids` is a small set; a
      DEPARTMENT_ADMIN's includes that department's child sections) — not
      just "login succeeds."
- [ ] Organizational unit count matches
      `database/sqlserver/seed/provisioning.v1.manifest.json`'s
      `record_counts.org_units_total` exactly (verified automatically by
      `verify_installation.sh` §6).
- [ ] User/account counts (total, active, inactive, by role) match the same
      manifest exactly (verified automatically by `verify_installation.sh` §6).
- [ ] Hospital Directory API settings page (Config → Hospital Directory
      API) loads without a 500 error, and Test Connection returns a real
      success/failure result (not a generic error).
- [ ] Table View loads and shows records (or an appropriately empty state
      on a fresh install).
- [ ] Dashboard Scope selector: selecting "Administration" then "Department"
      then "Section" shows real cascading options at each level (not empty
      dropdowns).
- [ ] Custom Table Views count (11) matches
      `provisioning.v1.manifest.json`'s `record_counts.custom_views_total`.
- [ ] Drawer Notes: the label picker shows at least the four seeded defaults
      ("Follow-up Required", "Resolved", "Escalated", "Internal Note").
- [ ] Patient History search and patient selection during complaint entry
      work after the Hospital Directory API integration is configured and
      saved (not just tested) — confirm via Config → Hospital Directory API
      that Save Settings (not just Test Connection) was clicked.
- [ ] ML Classification: submitting complaint text returns real predictions
      for domain, severity, harm level, stage, feedback type, improvement
      opportunity, and classification-EN (always), plus category and
      subcategory (for complaints that route to categories 3, 4, 5, 6, or 7).
      Complaints routing to category 1 or 2 are expected to show
      subcategory as "Not Available" — this is a known, documented model
      limitation (see `RELEASE_NOTES.md` and
      `ML_CLASSIFICATION_ISSUE_FOR_DEV_TEAM.md`), not a failure to
      investigate during this deployment. `scripts/qualify_offline_installation.sh`
      checks this automatically (domain + severity always present; category/
      subcategory not asserted pass/fail since the correct result depends
      on which category the input routes to).
- [ ] Model Dashboard (Settings → Training) shows a clear "No training runs
      yet" message on a fresh install, not raw "0/0/0%".
- [ ] Settings → Training → "Database Growth (Last 30 Days)" chart populates
      after real incidents are created and a training run completes (counts
      `ml.CaseTrainingRecord` in SQL Server now, not the old unshipped
      `patient_feedback_ml.db` SQLite file — expected empty on a brand-new
      install with no incidents yet, not a failure to investigate).
- [ ] Doctor/Worker search (Insert Record and History pages): a name that
      exists in both the reserve table and the Hospital Directory API
      appears exactly once, not duplicated.
- [ ] Doctor/Worker History page: profile, statistics, incidents, and export
      all load without error for both a locally-created (reserve) doctor/
      worker and one sourced only from the Hospital Directory API (never
      selected/materialized before) — the latter shows a valid empty/zero
      history rather than an error.
- [ ] Restart test: `docker restart pfms-backend` then re-run
      `./scripts/verify_installation.sh` — confirms the app recovers
      cleanly from a container restart.
- [ ] Stop/start test: `./scripts/stop_stack.sh` then
      `./scripts/start_stack.sh`, then re-verify — confirms data and
      configuration survive a full stack cycle.

## Known Limitations to Confirm the Operator Understands

- [ ] Operator has read `RELEASE_NOTES.md` → "Known Gaps".
- [ ] `assets/whisper-model-medium/` was extracted and confirmed non-empty
      before first install (Speech-to-Text runs fully offline from this
      asset — no Internet access is required for it).
- [ ] Operator knows where backups are stored (`backups/` in the release
      folder) and how to restore one (`BACKUP_RESTORE.md`).
- [ ] Operator knows how to view logs and check container health
      (`LINUX_COMMANDS_REFERENCE.md`).

## Sign-off

| Field | Value |
|---|---|
| Deployment date | |
| Application Release version | 1.0.0 |
| Deployed by | |
| Verification result | PASS / FAIL |
| Rollback tested | YES / NO |
| Notes | |
