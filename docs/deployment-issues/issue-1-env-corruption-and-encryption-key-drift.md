# Issue 1 — `.env` corruption cascading into DB outage + Hospital Directory settings failure

**Date:** 2026-09-28 → 2026-09-29 (OR-LAB)
**Status:** RESOLVED. DB connectivity restored, `SETTINGS_ENCRYPTION_KEY` fixed, all required `.env` credentials confirmed present and correct, application confirmed working end-to-end. One related issue remains open but is explicitly outside PFMS/Docker (see "Related, but explicitly not caused by tonight's work" below).
**Systems affected:** PFMS/HCAT (`/opt/pfms` on OR-LAB) — `sqlserver`, `backend`, `frontend`, `db-init`.

## Summary

An image update on the air-gapped OR-LAB system uncovered a `.env` file that had been silently corrupted over a week earlier. Recreating containers during the update caused them to pick up the corrupted file for the first time, breaking database connectivity. Recovery restored the database, but surfaced a second, unrelated corruption in a different secret used by the Hospital Directory API settings feature, and — one layer further in — a stale provisioning artifact that had drifted from the live database over time (written up separately as Issue 2, since it turned out to be a genuinely distinct root cause rather than more fallout from the `.env` corruption). No patient data was lost at any point, and the application is confirmed working end-to-end as of this writing.

## What triggered it

Updating PFMS with new backend/frontend images. The VM ran low on disk space mid-update; `docker image prune` was run to clear space, then the new images were loaded. Immediately after, Hospital Directory settings started failing. Force-recreating `frontend`/`backend` made things worse. Force-recreating `sqlserver` initially looked like it had deleted the database entirely.

**Worth stating plainly, since it's an easy wrong conclusion to draw from the timing alone: `docker image prune` did not cause the `.env` corruption, and has no mechanism to.** It only removes unused Docker image layers — it cannot touch a plain host file like `/opt/pfms/.env`, which lives entirely outside Docker's image store. The two are connected only by sequence, not causation: prune freed disk space, new images were then loaded, and *that* update required recreating containers to run the new images — and it's specifically container recreation (not prune) that made Docker Compose read `.env` fresh for the first time since Sep 18, finally exposing a corruption that had been sitting there, invisible, for over a week. See "Root cause" below for what actually caused the corruption itself.

## Root cause

`/opt/pfms/.env` had been silently emptied to 0 bytes on **Sep 18** — over a week before this incident — almost certainly from an old, unrelated editing session (a leftover `.env` swap file from the same few minutes corroborated this). It didn't matter until tonight because the already-running containers still had real credentials baked into their process environment from before Sep 18. The moment anything got recreated, Docker Compose re-read the now-empty `.env`, injected blank DB credentials into the new containers, and that broke every DB-dependent feature.

**Nothing was actually deleted.** `docker ps -a` showed `pfms-sqlserver` still running (just unhealthy), and `docker volume ls` confirmed `pfms_sqlserver_data` — where the real patient data lives — was fully intact the entire time. A locked door, not a demolished house.

## Recovery steps taken

1. Located a genuine backup, `.env.bak_20260824_070332` (Aug 24, predates the corruption).
2. Restored it as `.env`.
3. Recreated only the `sqlserver` container (`docker compose --env-file .env -f docker-compose.yml -p pfms up -d sqlserver`) — the data volume is untouched by a container recreate, only the container itself is replaced.
4. Planned next: recreate `backend`, then `frontend` (skipping `db-init`), matching the documented safe install-runbook pattern — deferred until the secondary issues below were understood, to avoid causing a second outage in the same night.

## Secondary problems found during recovery

These were found by verifying each step rather than assuming the restore was complete — several would have caused a repeat outage if skipped.

- **`DATABASE_USER` / `DATABASE_PASSWORD` came back unset** even after restoring the "good" `.env`. Confirmed these are genuinely required variables (present in `.env.example`, in the `1.2.0` release's own compose file, and consumed by `database/sqlserver/scripts/ensure_login_exists.py`) that the restored backup simply didn't include. **Resolved**: set to real values (`pfms_user` + a freshly chosen password, since `ensure_login_exists.py` is check-then-create and never alters an existing login, so a fresh password was safe to pick — this was in fact the *first* successful creation of that login on this instance, since every prior `db-init` run had exited before reaching it). `db-init` subsequently created the login, granted `db_owner`, and completed successfully.
- **`MSSQL_SA_PASSWORD` was present and valid immediately after the initial restore**, confirmed by the absence of a warning about it at that point — but was later found blank. A subsequent manual `.env` edit (there were several `nano .env` sessions during the incident) re-corrupted it a second time. **Resolved**: recovered via a working `sa` login found in an existing DBeaver connection profile, after first confirming that connection profile actually pointed at the correct database (it did not obviously match on port or database name at first glance — worth double-checking any "known good" saved connection against the actual system before trusting it). Confirmed correct via `sqlserver`'s own healthcheck (which authenticates as `sa` using this exact value) reporting `Healthy`.
- **`docker-compose.yml` itself was hand-edited multiple times** during the incident. A `diff` against a `.bak` copy found: an intentional `APP_VERSION` default bump (part of the update), plus newly added references to `MSSQL_SA_PASSWORD` / `DATABASE_USER` / `DATABASE_PASSWORD`. Live hand-editing of compose files mid-incident is exactly how this kind of drift gets introduced — worth diffing against a backup any time multiple edits happen in sequence under time pressure. (These specific additions turned out to be correct and necessary, not drift to revert — `db-init`'s `environment:` block genuinely needs all three.)
- **`sqlcmd` tooling friction**: `/opt/mssql-tools/bin/sqlcmd` does not exist in this image; the real binary is at `/opt/mssql-tools18/bin/sqlcmd`. The newer bundled `sqlcmd` (mssql-tools18) is strict about argument spacing — a password argument immediately followed by another flag with no space between them silently mis-parses as "Unknown Option" instead of erroring clearly — and defaults to strict TLS certificate verification, requiring an explicit `-C` to trust the container's self-signed certificate. Neither behavior matches the older `sqlcmd`, and both cost significant time before being identified.
- **`SETTINGS_ENCRYPTION_KEY` found malformed** in the restored `.env` — 45 characters instead of the 44 required for a valid Fernet key (a single stray trailing character). This is the confirmed root cause of a separate symptom: `GET /api/config/external-api` returning `500`, making the Hospital Directory API Settings page appear completely empty/broken. The underlying data was never actually lost — querying `dbo.APP_ExternalApiSettings` directly confirmed a real, previously-saved row (`BaseUrl`, an encrypted API key) was intact in the database the whole time; the backend just couldn't construct a valid decryption cipher to read it back out. **Resolved**: removing the stray trailing character was confirmed (via an isolated decrypt test against the actual stored ciphertext, run before touching `.env` for real) to be the *entire* bug — the original key was otherwise correct, and decrypted the stored value to the real, original Hospital Directory API key. No data was lost; no re-entry was needed.
- **`PASSWORD_EXPORT_ENCRYPTION_KEY`** (a separate, unrelated secret backing a reversible credential-export feature, distinct from the one-way bcrypt login hashes) was checked as a precaution given the pattern above and found well-formed — no issue there.

## Related, but explicitly not caused by tonight's work

The Hospital Directory API (`malin`, the hospital's external HIS system) is unreachable: `curl` gets `Connection refused` (not a timeout) from both the VM and the physical Air-Gapped PC, meaning the network path is fine but nothing is listening on the target port right now. This points to the HIS-side service being down or moved, not a PFMS/Docker configuration problem. Next action: ask whoever administers `malin` to confirm the `his-general` directory API is actually running.

Two things surfaced while investigating this that need a human answer, not more guessing:
- **Port discrepancy**: `malin:8080` (an independently confirmed real capture, and the value actually saved in the working settings row) vs. `malin:8000` (mentioned in tonight's incident notes). Not yet resolved.
- **False reassurance from existing tooling**: a PowerShell connectivity test that "looked like it passed" was actually silently testing the local mock server by default when no target was explicitly passed, not the real `malin` host — giving false confidence the real service was reachable when it had never actually been tested.

## Non-issue, resolved

Version tag confusion (`1.0.0` vs `1.2.3` vs `1.3.1`) — cosmetic only. All three tags point at the same image content; the real build was tagged `1.0.0` at build time and manually re-tagged on OR-LAB because `.env` (and therefore the expected `APP_VERSION`) was unreadable at the time credentials were being recovered. The running code has been correct throughout.

## Follow-up actions

Still open as of resolution — none of these blocked getting the application working again, but none have been actioned either:

1. `.env` has no integrity check today — a silently emptied or partially corrupted file isn't detected until something gets recreated. Consider a startup validation step that checks all required keys are present and well-formed (right length/format for the ones that have one, like Fernet keys) before allowing a service to start with blank/invalid credentials, rather than defaulting silently.
2. Back up `.env` on a real schedule, not opportunistically. The `.env.bak_20260824_070332` backup that made tonight's recovery possible predates the corruption by luck of timing — and even it turned out to be missing keys a fully current backup should have had. Now that `.env` has real, complete, verified values, take a fresh backup of it while it's known-good.
3. Connectivity-test tooling should never silently default to a mock/local target. Require an explicit target for anything run against a real environment, so a "passing" test can't quietly mean "tested the wrong thing."
4. Confirm the real `malin` port (`8080` vs `8000`) with whoever administers it, and correct whichever internal documentation currently has the wrong one. Still unresolved — this is external to PFMS and needs a human on the HIS side.
5. Decide deliberately whether the credential-export endpoints (currently under an `/api/admin/testing/` path, originally scoped as test-only) should be renamed/rescoped now that they're used for real production credential distribution — not urgent, but worth a conscious decision rather than leaving it as an accident of naming history.
6. The seed folder (`database/sqlserver/seed/`) had restrictive host-level permissions that blocked a straightforward file-manager copy-paste during recovery, requiring a temporary `chmod 777` (reverted afterward). Worth deciding on a less all-or-nothing way to authorize a one-off file replacement there in future.

See also **Issue 2** (`issue-2-provisioning-artifact-drift.md`) for a related but distinct problem found immediately after this one was resolved: a stale `provisioning.v1.json` that had drifted from the live database over time, unrelated to the `.env` corruption itself.
