# ML Architecture Decision Record — Operational/ML Data Consolidation & Bulk Import

**Status**: ACCEPTED — target architecture agreed. Implementation not yet started.
**Date**: 2026-07-16
**Supersedes**: the standalone-SQLite ML persistence design (`models_directory/patient_feedback_ml.db`) as the live production ML store.

---

## 1. Why this document exists

A request to "add ML processing to the Excel bulk-import feature" turned out to rest on a broken foundation. Before building anything, the system was investigated end-to-end (database inventory, ML component map, insert vs. import flow trace, and current database/Docker/migration posture). Those investigations, plus an independent architecture proposal and cross-review, converged on one target design. This document is the durable record of that decision — the reasoning, the schema, and the sequencing — so the "why" survives past this conversation.

---

## 2. Confirmed findings that drove this decision

These were verified against the live system (SQL Server query results, live SQLite schema introspection, source code), not assumed:

1. **Two databases, two engines, no relationship.** Operational data lives in SQL Server (`IncidentManager`, SQL Server 2022 Express, 16.0.4245.2). ML training data lives in a standalone SQLite file (`models_directory/patient_feedback_ml.db`, ~116 MB) inside the application source tree, with a hard-coded path.
2. **No shared identifier.** `patient_feedback_encoded.id` is a SQLite-local autoincrement with no relationship to `APP_IncidentCase.IncidentRequestCaseID`. No foreign key exists or is possible cross-engine.
3. **Manual insert calls the ML adapter; both live Excel-import pipelines do not.** `insert_service.create_record()`/`update_record()` call `ml_insert_adapter.add_corrected_record_to_ml()`. Neither `import_service.process_upload()` (`/api/import/upload`) nor `table_view_service.bulk_import_records_from_excel()` (`/api/complaints/import-excel`) call it at all.
4. **Editing a case does not update its ML record — it appends a new, unlinked one.** There is no update-in-place logic on the ML side; every edit creates another orphaned row.
5. **ML failures are silently swallowed.** Every ML write is wrapped in a bare `try/except` that only `print()`s. The operational record's success is never affected by, or even aware of, ML failure.
6. **Four independent case-creation implementations exist**, each duplicating validation/hierarchy/default logic with drift: `insert_service.create_record()`, `migration_insert_service.create_record_migrated()`, `import_db.insert_case()`, `table_view_service._insert_historical_record()`. Only the first two call the ML hook.
7. **The current adapter computes up to 11 embeddings per case, unbatched, one model call at a time** (`get_embedding()` called individually; the batch-capable `get_embedding_list()` exists but is unused). Of those 11, embedding-column-usage analysis across every `train_*.py` script in the repo found:
   - `embedding_text1` (complaint text alone) — used by nearly every trained model. **Confirmed load-bearing.**
   - `embedding_text123` (complaint+immediate+taken combined) — used by roughly half the trained models. **Confirmed load-bearing.**
   - `embedding_text2`, `embedding_text3`, `embedding_text23` — no confirmed consumer in any training script.
   - `sentence_1..6_embedding` — consumed only by an experimental Stage-model variant (`train_ML_Metric_Mapper_Numeric`); the production Stage classifier (`classify_stage_Score_Based`, marked `# The method Used` in its own source) does not read these columns at all.
8. **No processing-state model exists anywhere.** No `pending`/`processing`/`completed`/`failed` column on any table, main DB or ML DB.
9. **No embedding-model-version tracking exists.** The code calls the model "MPNet"; its actual saved config identifies it as `XLMRobertaModel`. Nothing ties a stored embedding to the model/version that produced it.
10. **Existing ML data cannot be assumed reproducible from the current main database.** Live counts: `APP_IncidentCase` = 173 rows; `patient_feedback_encoded` = 549 rows; `patient_feedback_encoded_Old` = 485 rows. The main DB has clearly been reset/cleared at some point (a `clear_all_complaints.sql` script exists) while the SQLite file was not. A large share of the ML rows likely correspond to no case that exists today — this is original, human-entered text and labels that exist **only** in that file.
11. **The separation is a historical accident, not a technical requirement.** No vector-index library is used anywhere (confirmed: no FAISS/Chroma/pgvector/Milvus/Qdrant/Annoy/HNSW in the repo). No SQLite-specific feature is exercised. Everything stored is plain relational data that SQL Server can hold natively.
12. **No Docker, no migration framework, no automated backup exist today.** No Dockerfile/Compose anywhere in the repo. No schema-version-tracking table. Backups of the main DB are manual (`C:\SQLBackup\*.bak`, ad hoc, no scheduled job found). The ML SQLite file has no backup process at all.
13. **A working precedent for real idempotency already exists**, just not applied to bulk import: `migration_service.migrate_legacy_case()` checks `APP_DataMigration_Map` (a table with a uniqueness constraint on `legacy_case_id`) before inserting, and is safely retry-able. Neither Excel-import pipeline does anything equivalent — re-uploading the same file creates full duplicates.
14. **Only one of the two Excel-import pipelines is reachable from the UI.** `/api/import/upload` is wired to `Front_End_Feedback_Analysis/src/components/settings/ImportTab.jsx`. `/api/complaints/import-excel` has a frontend wrapper function (`importExcel()` in `src/api/complaints.js`) that no component calls — confirmed dead from the UI's perspective, though still directly callable over HTTP.

---

## 3. Architectural principles (binding going forward)

1. **Operational reliability comes first.** Creating, viewing, assigning, publishing, or closing a case must never depend on ML processing succeeding.
2. **Every ML record has a known, traceable origin** — an operational case, a historical standalone example, a specific import batch, or a specific migration — never an unexplained row.
3. **All case-creation paths reuse one business service.** No future insertion path (manual, import, migration, external API) re-implements validation, hierarchy rules, defaults, or ML registration independently.
4. **ML processing is observable and recoverable.** Every unit of ML work has a persisted status (`Pending` / `Processing` / `Completed` / `Failed` / `RetryPending`) and a recorded error on failure. Nothing disappears into console output.
5. **Database data survives application replacement.** No live database may be stored only inside a source-code directory, a Docker image layer, an application container filesystem, or a release ZIP that gets replaced on update.

---

## 4. Decisions

### 4.1 Where should ML data live?

**Decision: an `ml` schema inside the existing `IncidentManager` SQL Server database — not a separate database, and not the current standalone SQLite file.**

| Criterion | Standalone SQLite (status quo) | Separate SQL Server database | `ml` schema, same database |
|---|---|---|---|
| Enforceable FK to `APP_IncidentCase` | Not supported | **Not supported** — SQL Server does not allow FK constraints across databases | Supported |
| Shared transaction with the case insert | Not supported | Supported (same instance, local transaction) | Supported |
| Backup/restore atomicity | Not supported (proven to drift — finding #10) | Two backups to coordinate | One backup, one restore |
| Docker volume/service complexity | Extra volume + hard-coded path to fix regardless | Extra database to provision | Simplest — one DB, one volume |
| Migration tooling | None exists; would need building from scratch | Extends existing `.sql` convention | Extends existing `.sql` convention |
| One-time migration effort | None (status quo) | Moderate ETL | Moderate ETL (same effort as separate DB) |
| Ongoing maintenance | Highest (two stores, proven drift) | Lower | Lowest |

The separate-database option does **not** get real FK enforcement — that was an error in an earlier draft of this analysis, corrected here: SQL Server has no cross-database FK support. That makes the `ml`-schema-in-the-same-database option the clear and decisive choice, not merely the tidiest one.

### 4.2 What does an ML record represent?

**Decision: one current record per *operationally-sourced* ML representation of a case.** Editing a case updates its existing ML record (labels and/or embeddings, as relevant) rather than appending a new one.

Refinement adopted during review: this rule governs records that originate from the operational system. It does **not** mean every valid training example must be linked to a case. The full training dataset may legitimately include historical complaints, legacy records, external datasets, or curated examples that have no operational case at all. Modeled as two tables with different integrity rules, rather than one table with a nullable FK:

- `ml.CaseTrainingRecord` — **required** FK to `dbo.APP_IncidentCase`, unique on `IncidentRequestCaseID`, one current row per case.
- `ml.HistoricalTrainingExample` — no case relationship required; holds standalone/legacy/unmatched training assets.

Full version-history tracking (every correction as its own row with `version_number`/`is_current`/`change_reason`) is a legitimate *future* feature, deliberately deferred — not something to half-build by accident again.

### 4.3 Should embedding generation block case creation or import?

**Decision: no.** Cases (manual or imported) are committed as operational records first. ML work is registered as a job and processed afterward by an asynchronous worker. The HTTP import request returns once operational records are created and jobs are registered — it does not wait for embeddings.

### 4.4 What worker technology?

**Decision: a SQL-backed job table (`ml.EmbeddingProcessingJob`) plus a single in-process background worker, running inside the existing FastAPI backend.** No Redis/Celery/RabbitMQ at this stage — this is a single-VM, offline-installed system currently holding a few hundred to low-thousands of rows; that infrastructure would solve a scale problem this system doesn't have, at the cost of new moving parts an offline hospital deployment has to keep alive without internet access. The job table (not in-memory state) is what makes the worker safely restartable — the codebase already has a working precedent for a persistent-retry background task in `backend/core/bootstrap.py`'s `startup_db_reconnect_watcher`.

If/when the system is Dockerized, the same worker logic can move into its own container without any database redesign — the job table already makes it stateless from the process's point of view.

### 4.5 Which embeddings should actively be generated going forward?

**Decision: `embedding_text1` (complaint text) and `embedding_text123` (complaint + immediate + taken, combined) only, for new/future records.** All 11 original columns are preserved for existing historical data during migration (see §4.7) — nothing is discarded, but only these two are computed going forward, pending the one remaining confirmation below.

**Open confirmation before this is frozen**: verify that `classify_stage_Score_Based()` (not the `Internal_metric_training_functions.train_ML_Metric_Mapper_Numeric()` variant) is what's actually wired into the live `package_models.classify_feedback()` call path. If confirmed, the sentence-level embeddings have no production consumer and the 9-of-11 trim is fully justified. This is a code-reading confirmation, not a re-investigation.

Effect: a 1,000-case import requires on the order of **~2,000** model forward passes going forward (2 embeddings × 1,000 cases), not the ~11,000 the original unbatched, all-11-columns design would require.

### 4.6 Which import pipeline survives?

**Decision: `/api/import/upload` is the official pipeline** (confirmed live in the UI, richer structure — incident grouping, target departments, doctor/worker linkage, draft subcases, rejected-row reporting, an existing smoke test). `/api/complaints/import-excel` is confirmed unreachable from the current frontend (dead wrapper function, no caller) but is deprecated, not deleted immediately — confirm with the team/check for any direct external caller before removal.

### 4.7 How is existing ML data preserved?

**Decision**: treated as a controlled migration project, not a routine schema change.

- Freeze and back up both stores first (fresh SQL Server `.bak`, checksummed copy of the SQLite file) before touching anything.
- Classify every SQLite row by match confidence against the current main DB: `Exact` / `High` / `Possible` / `Unmatched` / `Conflict`. Only `Exact`/`High` matches are auto-linked into `ml.CaseTrainingRecord`. Everything else — including all of `patient_feedback_encoded_Old` — becomes `ml.HistoricalTrainingExample`, preserved with its original text, labels, embeddings, source table/row ID, and a note, without a forced case link.
- `table_feedback_train`/`table_feedback_test` are **not** migrated as authoritative tables — they're derived from `patient_feedback_encoded` via `split_data.py`. Preserve their row counts for audit, and regenerate them later using a versioned, recorded split method/seed.
- Verify before retiring anything: source/destination row counts, non-null embedding counts, text and label spot-checks. The original SQLite file is archived read-only after sign-off — never deleted as part of the same step that migrates it.

### 4.8 Duplicate prevention for import

**Decision: generalize the existing `APP_DataMigration_Map` pattern rather than invent a parallel mechanism.** Concretely:
- Batch-level: a file checksum recorded on an `ml.ImportBatch` row, to catch exact re-uploads.
- Record-level: an `ImportSourceRecordMap`-style table (`ImportBatchID`, `ExternalSourceSystem`, `ExternalRecordID`, `IncidentRequestCaseID`, `ImportedAt`) with a uniqueness constraint — the same shape of protection `APP_DataMigration_Map` already proves works for the Phase K legacy-migration path.
- In-file: detect duplicate rows/group keys before insertion, as today's `import_service._validate_group()` partially does.
- Where no stable external ID exists in the source data (today's template has none), a normalized fingerprint (text + date + patient + department) can warn rather than silently reject, pending a product decision on how strict this needs to be.

---

## 5. Target architecture

```
Frontend
   │
   ├── Manual Insert
   └── Bulk Import
          │
          ▼
Central Case Creation Service
          │
          ├── Validate (shared rules, all sources)
          ├── Create operational case + related tables
          ├── Apply source-specific policy (FSM/workflow/duplicate/publication)
          └── Register ml.EmbeddingProcessingJob (same transaction as the case insert)
                         │
                         ▼
        SQL Server — "IncidentManager" database
        ┌───────────────────────────────┬───────────────────────────────┐
        │   dbo schema (operational)    │   ml schema (ML concern)       │
        │   APP_Incident                │   CaseTrainingRecord           │
        │   APP_IncidentCase            │   HistoricalTrainingExample    │
        │   workflow/lookup tables      │   EmbeddingProcessingJob       │
        │                               │   EmbeddingModelVersion        │
        │                               │   ImportBatch / SourceRecordMap│
        └───────────────────────────────┴───────────────────────────────┘
                         │
                         ▼
                ML Processing Worker (in-process, backend)
                         │
                         ├── Claim pending jobs (batched)
                         ├── Batch-generate embedding_text1 / embedding_text123
                         ├── Upsert ml.CaseTrainingRecord (one current row per case)
                         ├── Record EmbeddingModelVersion used
                         └── Mark Completed / Failed(+retry policy)
```

---

## 6. Logical data model

*(Exact SQL types finalized at implementation time — this is the agreed shape.)*

**`ml.CaseTrainingRecord`** — current ML representation of an operational case.
`CaseTrainingRecordID`, `IncidentRequestCaseID` (FK, UNIQUE), `ComplaintText`, `ImmediateActionText`, `TakenActionText`, label columns (`FeedbackTypeID`…`ImprovementOpportunityTypeID`), `ComplaintEmbedding`, `CombinedTextEmbedding`, `EmbeddingModelVersionID`, `EmbeddingDimension`, `ProcessingStatus`, `LastProcessedAt`, `SourceDataUpdatedAt`, `CreatedAt`, `UpdatedAt`.

**`ml.HistoricalTrainingExample`** — preserved legacy/unmatched training data, no case FK required.
`HistoricalTrainingExampleID`, `LegacySource`, `LegacySourceTable`, `LegacySourceRowID`, `PossibleIncidentRequestCaseID` (nullable), `LinkConfidence` (`Exact`/`High`/`Possible`/`Unmatched`/`Conflict`), original text/labels, **all original embedding columns preserved as authored**, `ImportedAt`, `MigrationBatchID`, `PreservationNotes`.

**`ml.EmbeddingProcessingJob`** — durable, retryable processing queue.
`EmbeddingProcessingJobID`, `IncidentRequestCaseID`, `JobType` (`Create`/`Reprocess`/`TextChanged`/`LabelsChanged`/`ModelUpgrade`/`MigrationBackfill`), `Status` (`Pending`/`Processing`/`Completed`/`Failed`/`RetryPending`/`Cancelled`), `AttemptCount`, `MaximumAttempts`, `RequestedAt`, `StartedAt`, `CompletedAt`, `NextRetryAt`, `LastErrorCode`, `LastErrorMessage`, `WorkerID`, `EmbeddingModelVersionID`, `ImportBatchID`.

**`ml.EmbeddingModelVersion`** — never trust a label alone (the "MPNet"/XLM-RoBERTa mismatch is exactly why).
`EmbeddingModelVersionID`, `ModelName`, `ModelPathOrIdentifier`, `ModelArchitecture`, `ModelChecksum`, `EmbeddingDimension`, `PoolingMethod`, `NormalizationMethod`, `TokenizerIdentifier`, `ActivatedAt`, `RetiredAt`, `IsActive`, `ConfigurationJson`.

**`ml.ImportBatch`** — one row per uploaded file.
`ImportBatchID`, `OriginalFileName`, `FileChecksum`, `TemplateVersion`, `UploadedByUserID`, `UploadedAt`, `Status`, `TotalRows`, `AcceptedRows`, `RejectedRows`, `DuplicateRows`, `CreatedCaseCount`, `MLCompletedCount`, `MLFailedCount`, `CompletedAt`.

**`ml.ImportSourceRecordMap`** — record-level idempotency, generalized from `APP_DataMigration_Map`.
`ImportSourceRecordMapID`, `ImportBatchID`, `ExternalSourceSystem`, `ExternalRecordID`, `IncidentRequestCaseID`, `ImportedAt` — unique on `(ExternalSourceSystem, ExternalRecordID)`.

**`dbo.SchemaMigrationHistory`** — the schema-version tracking that does not currently exist anywhere in this project.
`MigrationID`, `MigrationName`, `Checksum`, `AppliedAt`, `AppliedBy`, `ApplicationVersion`, `Success`.

---

## 7. Edit and workflow behavior

- **Text change** (`complaint_text`/`immediate_action`/`taken_action`) → registers a `TextChanged` job; worker recomputes the two active embeddings and updates the one current `ml.CaseTrainingRecord` row. No duplicate row is created.
- **Label-only change** (domain/category/severity/etc.) → labels updated directly; no embedding recompute needed since source text didn't change.
- **Unrelated change** (workflow assignment, publication state) → no ML job at all.
- **Publication/workflow is never gated on ML status.** A case may be `Operational Status: Open` / `ML Processing Status: Pending` simultaneously — that is a valid, expected state, not an error condition.

## 8. Failure and retry

- Recoverable failures (transient DB error, model temporarily unavailable, backend restart mid-job) → increment `AttemptCount`, store the error, move to `RetryPending` with a `NextRetryAt`.
- Permanent failures (missing source text, dimension mismatch, max attempts reached) → `Failed`, case remains fully operational regardless, failure surfaced for manual attention/retry.
- On backend startup: sweep jobs stuck in `Processing` beyond a threshold back to `RetryPending` — the job table, not in-memory state, is always the source of truth.

## 9. Docker and offline-deployment requirements (for when Docker is introduced)

- **Database data must never live only inside an ephemeral application container** — today's biggest concrete version of this risk is the SQLite file sitting inside the source tree; a naive `COPY . .` Dockerfile would bake 116 MB of irreplaceable historical text/labels into an image layer and silently reset it on every container replacement. This consolidation plan removes that risk structurally by moving the live store into SQL Server's own persistent volume.
- One SQL Server service/volume holds both `dbo` and `ml` — one backup, one restore, one volume.
- The ~1.1 GB model-artifact directory (`models_directory/Classification_Models/model_storage/`) needs an explicit packaging decision (baked into the image, read-only mounted volume, or versioned bundle imported at install time) with its checksum tracked in `ml.EmbeddingModelVersion` — not left to whatever happens to be in the image at build time.
- Startup ordering/health checks build on the existing `bootstrap.py` DB-reconnect-watcher pattern already in the codebase.
- Update procedure: verify release checksum → back up DB → apply ordered migrations → load model artifacts → register model version → replace images → health check → verify schema version → verify worker operation → roll back from backup if any step fails.

---

## 10. Priority tiers

### Required now (fixes the actual import/data-integrity problem)
- Back up and inventory the existing SQLite data.
- Create the `ml` schema in `IncidentManager`.
- Define `ml.CaseTrainingRecord`, `ml.EmbeddingProcessingJob`, `ml.EmbeddingModelVersion`.
- Centralized case-creation service (collapses the four duplicated implementations).
- Official import flow (`/api/import/upload`) registers ML jobs instead of bypassing ML entirely.
- Asynchronous, batched embedding processing — HTTP import request no longer waits on ML.
- Persisted failure/retry, replacing silent `print()`-only failure.
- Stop appending duplicate ML rows on edit — update the one current record instead.
- Import idempotency via the generalized `ImportSourceRecordMap` pattern.

### Required before retiring SQLite / before Docker packaging
- Historical-data migration tooling and execution (`ml.HistoricalTrainingExample`, match-confidence classification).
- `dbo.SchemaMigrationHistory` and a real migration convention.
- Backup/restore verification procedure.
- Model-artifact packaging + checksum strategy.
- Removal of the hard-coded ML DB path.

### Valuable later (does not block the core architecture)
- Full ML-administration dashboard.
- Fine-grained per-action ML permissions.
- Dedicated worker container (separate from the backend process).
- Full version-history tracking of every case correction (Option B from §4.2, deliberately deferred).
- Advanced import monitoring, multi-user concurrency controls beyond essential DB-level protection.

---

## 11. Open confirmations before schema is frozen (Phase 0)

1. Confirm `classify_stage_Score_Based()` (not the metric-mapper variant) is the live production Stage classifier — settles whether the 9 non-essential embedding columns can be dropped from active computation with full confidence.
2. Confirm whether `/api/complaints/import-excel` has any direct external caller before deprecating/removing it.
3. Confirm expected real-world import batch sizes, to size worker batch configuration sensibly.
4. Confirm whether SQL Server Express remains the target edition long-term (relevant to the 10 GB per-database cap as ML data grows).
5. Confirm whether correction-history tracking (§4.2's deferred Option B) is wanted as a near-term or long-term item, so it's a known deferral rather than a silently dropped idea.

None of these change the target architecture in this document if answered differently than expected — they only tune specifics within it.

---

## 12. Sign-off

Architecture direction agreed via joint investigation, independent proposal, and cross-review. Recorded here as the reference document for implementation planning.

---

## 13. SQLite Retirement Record (Stage 13, completed 2026-07-20)

The legacy ML SQLite store (`models_directory/patient_feedback_ml.db`, ~116 MB) is officially retired. All valuable data was migrated into SQL Server's `ml` schema across Stages 1-11; Stage 12 proved migration resumability, multi-job worker recovery, training-run crash recovery, and `.bak` restore all work before this final cutover.

**Archive**: a final checksummed, read-only copy was taken immediately before removal — `C:\SQLBackup\ml_stage13_final_archive_20260720_123654\patient_feedback_ml.db`, SHA-256 `2d7ccbc89a3951db0b764a75ec2d23af624e623f17e9371861aa8aee784c0851`, manifest at `stage13_final_archive_manifest.json` in the same folder. This sits alongside the Stage 1 (`ml_stage1_archive_20260716_124546`) and Stage 8 (`ml_stage8_freeze_baseline_20260716_161151`) archives as historical/emergency-recovery evidence.

**Two live leaks closed during this stage** (found only because Stage 13's removal-scope inventory went file-by-file, not assumed clean from Stage 5/8's "freeze"):
1. `backend/api/services/migration_insert_service.py` still called the legacy `ml_mapping.add_corrected_record_to_ml()` SQLite hook on every legacy-case migration — Stage 8's freeze only removed `case_service.py`'s call site. Replaced with the same `ml.CaseTrainingRecord` upsert + `ml.EmbeddingProcessingJob` registration `case_service.create_case()` already does, so migrated cases remain training-eligible.
2. `backend/api/services/training_service.py`'s real "Train Now" pipeline (`_run_split_data()`) was calling the legacy, un-deduplicated `split_data()` instead of `split_data_from_sql_server()` — meaning every live training run was silently reading the old, un-deduplicated dataset, undoing the duplicate/test-fixture cleanup done earlier in this project. Repointed to `split_data_from_sql_server()`; verified via a real training run that `Harm_Binary`'s record count (387) now matches the known-correct deduplicated train split.

**`ml_mapping/ml_insert_adapter.py` deleted entirely** (478 lines, pure legacy SQLite write logic), along with its re-export from `ml_mapping/__init__.py`. Its only remaining importers were ~12 ad-hoc, already-dead debug/verification scripts loose at the `backend/` root (`final_verification.py`, `verify_implementation.py`, `TEST_FOUR_FIELDS.py`, etc.) — none wired into the live app; breaking them is an accepted, intentional consequence of retirement, not a regression.

**Design decision (confirmed with user)**: rather than rewriting the ~20 existing `train_*.py` trainer scripts off SQLite entirely, `split_data_from_sql_server()` continues to materialize `table_feedback_train`/`table_feedback_test` into a small SQLite file at the same path — but now purely as an **ephemeral hand-off format**, fully regenerated fresh (`if_exists="replace"`) on every split, never accumulating data, never containing anything but the current split. This is a fundamentally different thing from the retired 116 MB accumulating live store, and satisfies the retirement gate ("no production code path reads from or writes to the *original* legacy database") without a 20-file rewrite. Confirmed via a rename-then-test drill: with the original file entirely absent, a real `split_data_from_sql_server()` + `train_all()` run regenerates a correct ~4 MB replacement from scratch and all 18 trainers succeed identically to before (same 3 pre-existing, unrelated failures — `Domain_Model`'s NULL-domain crash, one of `Category_Domain1/3`'s or `Subcategory_Cat6`'s index-out-of-bounds, varying slightly run to run with the random split composition).

**`get_current_ml_db_size()`** (in `backend/api/db_layer/training_db.py`) already returned `0` gracefully when the file is missing, and its only caller (`training_service.py`) already skips recording rather than writing garbage — no code change was needed here; the `ml_db_size_history` metric simply stops growing, frozen at its last real value.

**`project_paths.py`'s `get_db_path()`** and the individual trainers' own `sqlite3.connect(DB_PATH)` calls are intentionally left pointed at the same physical path — see the ephemeral-hand-off decision above.

Workstream 2 (Stages 8-13) is now fully complete.
