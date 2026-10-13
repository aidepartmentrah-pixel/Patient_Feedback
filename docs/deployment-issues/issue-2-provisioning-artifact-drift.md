# Issue 2 — `provisioning.v1.json` had drifted from the live database

**Date:** 2026-09-29 (OR-LAB)
**Status:** RESOLVED.
**Systems affected:** PFMS/HCAT (`/opt/pfms` on OR-LAB) — `db-init` (`provision.py`), specifically `dbo.AdminsrationUnit` and `dbo.APP_CUSTOM_VIEWS`.

## Summary

Found immediately after Issue 1 was resolved, while recreating `backend` for the first time in this incident (which pulls in `db-init` as a dependency). Distinct root cause from Issue 1 — not more `.env` fallout. `db-init`'s provisioning step (`provision.py`) compares the live database against a build-time artifact (`provisioning.v1.json`) and refuses to proceed if any record disagrees, by design ("no table's rerun path reports success while silently ignoring a real mismatch"). That artifact had genuinely drifted from the live database in two places, both because live-database fixes made directly against production, at some point in the past, were never backported into the artifact that generates fresh installs/re-provisioning.

## How it surfaced

`provision.py`'s drift check raises on the *first* mismatching record it finds per table and stops — it does not collect or report every mismatch in one pass. This meant the fix happened in two rounds, one table at a time:

1. First attempt: `db-init` failed immediately on `dbo.AdminsrationUnit` (org units), record 20 ("المحاسبة العامة"): live database had `Type=324`, the artifact said `Type=325`.
2. After that table was fixed and `db-init` re-run: it passed org units cleanly, then failed on `dbo.APP_CUSTOM_VIEWS`, view 1036 ("Classifications"): a `ShowSectionAnswer` display flag disagreed.

## Root cause

`324`/`325` are org-hierarchy level constants (`SECTION`/`DEPARTMENT`, not the complaint-classification taxonomy that a similarly-named seed script also touches — confirmed by reading `backend/api/constants/org_unit_types.py` directly, not assumed from the numbers alone). At some point before this incident, a batch of organizational units had been found wrongly classified as `DEPARTMENT` when their real role-scope usage expected `SECTION`, and were corrected directly against the live database. `provisioning.v1.json` — a build artifact generated from a raw extract off the legacy system — was never regenerated after that live fix, so it still carried the pre-fix values. The custom-view display-flag drift was the same shape: a live correction that never made it back into the artifact.

**This is not a data-loss issue and never was.** The live database held the *correct* values in both cases; the artifact was the stale one. `provision.py`'s strict compare-and-refuse behavior did exactly its job — it caught real drift instead of silently overwriting correct live data with a stale snapshot, which is the actively dangerous alternative it was deliberately designed to avoid.

## How it was diagnosed and fixed

Given the risk of hand-transcribing data from a live production database under time pressure, each drift was resolved by pulling the *complete* comparison as structured data rather than trusting one-at-a-time error messages or photographed terminal scrollback:

- **Org units**: a SQL query joined the live `dbo.AdminsrationUnit` table against a `VALUES` table built directly from `provisioning.v1.json`'s 175 `org_units` entries (`source_id`, `name`, `parent_source_id`, `type` — all three fields `provision.py` actually compares, not just `Type`, after an initial narrower version of this query was corrected to include `Name` and `ParentID` too). This returned all 55 actually-mismatched units in one pass, all following the identical pattern (`Name`/`ParentID` already correct, only `Type` stale at `325` instead of `324`) — confirming a single coherent batch, not scattered unrelated corruption.
- **Custom views**: a smaller, 11-row table, all 49 boolean display-flag columns plus `ViewName`/`IsActive` extracted as a plain CSV via `sqlcmd` and diffed locally against the JSON — this surfaced exactly 2 real differences (`ShowSectionAnswer` on view 1036, `ShowClassificationID` on view 1034).
- **The query itself was tested before being trusted**: an early version, structured as a `WITH ... AS (VALUES ...)` CTE, failed with a genuine T-SQL syntax error — reproduced against a disposable local SQL Server container (not just assumed to be a DBeaver-specific quirk), which confirmed the bug was real and not tool-dependent. Rewritten as a plain `JOIN (VALUES ...)` derived table instead, then verified against a small hand-built test table (including the real known-mismatched record, a deliberately-mismatched control record, and a NULL-handling edge case) before being run against the real database.
- Both `provisioning.v1.json` patches were applied programmatically (not hand-edited), and each patch was verified afterward to have changed *only* the intended fields — confirmed by structurally diffing the full before/after JSON, not just spot-checking the records that were supposed to change.
- `provisioning.v1.json.sha256` was regenerated to match, since `provision.py` gates the entire run on this checksum (`load_and_verify_artifact()`) — a stale checksum would have blocked the fix from ever being read, checksum mismatch or not.
- `provisioning.v1.manifest.json` needed no change — `provision.py` only checks that it *exists*, not its own checksum against the artifact.

## Confirmed safe: read-only until the actual fix

The diagnostic queries used to find the drift were plain `SELECT` statements — no data was modified by running them, regardless of how many attempts it took to get the query syntax right. The only writes were the deliberate, verified `provisioning.v1.json` patches themselves, applied once the full, real mismatch list was known — not a guess acted on early.

## Follow-up actions

1. When a live-database correction is made directly against production (as apparently happened for the org-unit batch and the custom-view flags, at some point before this incident), regenerate `provisioning.v1.json` (and its checksum/manifest) from the corrected state at the same time — otherwise the artifact silently drifts and the next `db-init` run rediscovers the same class of problem.
2. Consider having `provision.py` collect and report *all* drift for a table in one pass instead of raising on the first mismatch — the one-at-a-time behavior is what made this incident take two rounds instead of one, and cost real time under pressure re-running `db-init` between fixes.
3. A "regenerate provisioning artifact matches live DB" check could reasonably run as part of the release process, catching this class of drift before it ships rather than at install/upgrade time on an air-gapped system where iterating is slow and costly.
