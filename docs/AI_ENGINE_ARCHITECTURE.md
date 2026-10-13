# The Patient Feedback AI Engine — Architecture Reference

**Purpose of this document**: a complete, source-verified walkthrough of how the ML/AI engine in this repository works — where training data lives, how preprocessing and embedding happen, how the 18 models are trained, how a user triggers retraining from the UI, how long it takes, and which libraries/technologies are involved. Written so the same design can be lifted into a new project. Every claim below is backed by a specific file in this repo (paths given throughout) rather than assumed.

---

## 1. What the engine actually does

On every new patient-feedback case (a complaint, compliment, or suggestion, usually in Arabic), the engine predicts **8 structured fields** from the free-text complaint:

| Field | Example values | Model family |
|---|---|---|
| Domain | Clinical / Management / Relational | Hierarchical (level 1) |
| Category | Communication, Safety, Quality of Care, ... (7 values) | Hierarchical (level 2, depends on domain) |
| Sub-category | Delays, Misinformation, Infection Control, ... (27 values) | Hierarchical (level 3, depends on category) |
| Severity | Low / Medium / High | Ordinal regression |
| Harm level | No Harm → Minor → Moderate → Severe → Death | Binary + 2 ordinal models |
| Stage | Admission, Care on Ward, Discharge, ... | Score-based (not a trained classifier — see §6.6) |
| Feedback type | Complaint / Compliment / Suggestion / Other | Ordinal classifier |
| Improvement opportunity type | Ordinary / Red Flag / Never Event | Ordinal classifier |
| Classification (EN) | One of ~76 fine-grained legacy classification codes | Multiclass classifier |

This runs both as a **live suggestion** at the moment a user is entering a case (Insert Record page) and as the thing being retrained by the **Settings → Training** tab's "Train All Models" button.

---

## 2. System overview

```mermaid
flowchart TD
    A[Case created / edited<br/>manual insert or Excel import] --> B[Central case-creation service]
    B --> C[(SQL Server: IncidentManager DB<br/>dbo schema — operational data)]
    B --> D[ml.EmbeddingProcessingJob<br/>queued in same transaction]
    D --> E[Embedding Worker<br/>background thread, polls every 10s]
    E --> F[Local embedding model<br/>XLM-RoBERTa 'MPNet', 768-dim, mean-pooled]
    F --> G[(ml.CaseTrainingRecord /<br/>ml.HistoricalTrainingExample)]
    G --> H["Train All Models" button<br/>Settings > Training tab]
    H --> I[split_data_from_sql_server<br/>80/20 dedup split]
    I --> J[(SQLite hand-off file<br/>table_feedback_train / test)]
    J --> K[train_all.py<br/>18 trainers, sequential]
    K --> L[Versioned run folder<br/>training_runs/{run_id}/]
    L -.explicit promote_run only.-> M[Live inference .pkl files<br/>loaded once at server startup]
    M --> N[classify_feedback<br/>used on every new case]
```

---

## 3. Where the ML data lives

### 3.1 Today (current architecture)
All ML data lives **inside the same SQL Server database as the operational data** (`IncidentManager`), in a dedicated `ml` schema — not a separate database, not a file on disk. This was a deliberate architectural decision, documented in this file.

Key tables:
- **`ml.CaseTrainingRecord`** — one current row per operationally-sourced case. Holds the complaint/immediate-action/taken-action text, all label columns, both active embeddings, `ProcessingStatus` (`Pending`/`Processing`/`Completed`/`Failed`/`RetryPending`), and a link to the embedding model version that produced its vectors. Editing a case's text re-queues a job and **updates this same row** — it never appends a duplicate.
- **`ml.HistoricalTrainingExample`** — legacy/standalone training examples with no live case behind them (old spreadsheets, migrated legacy data). No FK required.
- **`ml.EmbeddingProcessingJob`** — a durable, retryable work queue (`JobType`: Create/Reprocess/TextChanged/LabelsChanged/ModelUpgrade/MigrationBackfill; `Status` with attempt counts and `NextRetryAt`). This is what makes embedding computation asynchronous and crash-recoverable.
- **`ml.EmbeddingModelVersion`** — records which literal model produced which embeddings (name, path, architecture, dimension, pooling method) — critical because embeddings from two different model versions are not comparable.
- **`ml.ImportBatch`** / **`ml.ImportSourceRecordMap`** — duplicate-prevention bookkeeping for bulk Excel imports.

### 3.2 Why not a vector database?
No vector-index library (FAISS/Chroma/pgvector/Milvus) is used anywhere in this codebase. Embeddings are stored as plain `VARBINARY`/BLOB columns (raw float32 bytes) and read back with `np.frombuffer(...)`. At this scale (hundreds to low-thousands of rows) a full relational table is simpler and gets transactional/FK guarantees a vector store wouldn't add here.

### 3.3 Historical note — what came before
Originally, ALL ML data (~116 MB) lived in a **standalone SQLite file inside the source tree** (`models_directory/patient_feedback_ml.db`) with a hard-coded path, completely disconnected from the operational SQL Server database (no shared ID, no FK, editing a case created a new orphaned ML row instead of updating one, and failures were swallowed by a bare `except: print(...)`). This was retired in July 2026 (see ADR §13). **If porting this design to a new project, skip this stage entirely and start with ML data inside the main relational database from day one.**

### 3.4 The SQLite file that still exists today
`split_data_from_sql_server()` (in [`models_directory/split_data.py`](../models_directory/split_data.py)) still materializes a small SQLite file at the same historical path — but purely as an **ephemeral hand-off format** between the SQL Server query and the ~20 existing `train_*.py` scripts, which all expect to read `table_feedback_train`/`table_feedback_test` tables. It is fully regenerated (`if_exists="replace"`) on every split and never accumulates data. This was a pragmatic choice to avoid rewriting 20 trainer scripts rather than a recommended pattern — **in a fresh project, have the trainers read directly from the real database.**

---

## 4. The embedding model (the core NLP component)

- **What it's called in code**: `"MPNet"` / `mpnet_embeddings`, matching `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` in naming.
- **What it actually is**: its own saved HuggingFace config identifies the architecture as **`XLMRobertaModel`** — a multilingual XLM-RoBERTa checkpoint, not literally MPNet. This mismatch is exactly why `ml.EmbeddingModelVersion` exists: never trust a label alone, always record the architecture from the model's own config.
- **Where it's stored**: `models_directory/Classification_Models/model_storage/mpnet_embeddings/` — a full local HuggingFace model folder (tokenizer + weights), **1.1 GB on disk**, loaded fully offline via `local_files_only=True` (no internet access needed or attempted at runtime — important for the hospital's air-gapped deployment).
- **Embedding dimension**: 768.
- **Pooling**: mean pooling over `last_hidden_state` across the token dimension (not the `[CLS]` token, not a `sentence-transformers` pooling head — plain PyTorch mean).
- **Tokenization**: standard HF `AutoTokenizer`, `truncation=True`, `max_length=512`, dynamic padding.
- **Loaded once per process**: module-level `_TOKENIZER`/`_MODEL` globals in [`models_directory/Classification_Models/Stage/modular_functions.py`](../models_directory/Classification_Models/Stage/modular_functions.py) (`get_embedding` / `get_embedding_list`). The FastAPI app forces this load at startup via a dummy `classify_feedback()` call (`startup_ml_warmup` in `backend/main.py`) specifically **to eliminate first-request cold-start latency** — without it, the first real classification after a server restart would pay the full model-load cost inline.
- **Batching**: `get_embedding_list()` tokenizes and forward-passes a whole list of texts in one batch (this exists specifically to avoid the old design's one-model-call-per-text bottleneck — see §9).

Two embeddings are computed **per case**, going forward:
1. `embedding_text1` — the complaint text alone.
2. `embedding_text123` — complaint + immediate action + taken action, concatenated.

(Nine other embedding-column variants exist in the schema for historical/experimental reasons but have no confirmed production consumer — see ADR §2 finding 7 and §4.5. **Don't carry that sprawl into a new project** — decide up front exactly which embeddings you need and only compute those.)

---

## 5. Preprocessing pipeline

Preprocessing is intentionally minimal — almost all of the "understanding" is delegated to the pretrained multilingual embedding model rather than hand-built feature engineering:

1. **Text assembly**: `complaint_text`, and `f"{complaint_text} {immediate_action} {taken_action}".strip()` for the combined variant. No stemming, no stopword removal, no manual tokenization before embedding — the transformer's own tokenizer handles that.
2. **Embedding**: as described in §4 — this is effectively the feature-extraction step. Everything downstream (LogReg/RandomForest/XGBoost/ordinal models) trains on the resulting 768-dim dense vectors, not on raw text or TF-IDF.
3. **Arabic-specific normalization utility** (`clean_arabic()` in `modular_functions.py`) exists — strips Arabic diacritics (tashkeel), normalizes variant letterforms (أ/إ/آ→ا, ى→ي, ة→ه, ؤ→و, ئ→ي), and strips non-Arabic characters/punctuation. This is used by the **Stage** classifier's vocabulary-matching logic (see §6.6), not by the embedding path — the embedding model consumes raw text as-is since a multilingual transformer tokenizer doesn't need hand normalization the way a bag-of-words model would.
4. **Sentence splitting** (`split_arabic_text_into_sentences`, a regex-based Arabic clause splitter, plus a Stanza-NLP-based ML variant and a punctuation/date-aware rules variant) exists for the experimental sentence-level "metric mapper" (an alternate Stage-scoring approach) but is **not** part of the live production path (`classify_stage_Score_Based` is confirmed as the actual production Stage classifier per ADR §11 item 1) — kept here as a worked example if per-sentence analysis is ever needed.
5. **Label encoding**: `sklearn.preprocessing.LabelEncoder` per model, with an explicit saved `label_map_*.json` mapping the encoder's local 0..n-1 space back to the real, stable database IDs — done because label sets aren't guaranteed contiguous or zero-based in the source data (e.g. domain IDs are 1/2/3, not 0/1/2).
6. **Train/test split**: `sklearn.model_selection.train_test_split`, **80/20**, `random_state=42`, with deduplication by `complaint_text` via a SQL `ROW_NUMBER() OVER (PARTITION BY complaint_text ...)` before the split — added specifically because the same complaint had been found duplicated across two legacy tables, which would otherwise let the same example leak into both train and test and silently inflate reported accuracy. Full details and rationale: [`models_directory/split_data.py`](../models_directory/split_data.py) `_fetch_sql_server_training_dataframe()`.
7. **Class imbalance handling**: partly manual (hand-tuned `class_weight` dicts per class in some trainers, e.g. Domain), partly left to each algorithm's own `class_weight='balanced'`-style options where used. No SMOTE/oversampling is used anywhere in this codebase.

---

## 6. The models — architecture per family

There are **18 trained model artifacts total**, run as one pipeline. Trainer scripts live under `models_directory/Classification_Models/` (production, versioned) with an earlier draft/reference copy under `model_training/` (not what's actually wired into the live app).

### 6.1 Hierarchical classification (11 models: Domain, 3×Category, 7×Subcategory)
For **each node** in the domain→category→subcategory tree, **three candidate algorithms are trained side by side**:
- `LogisticRegression(max_iter=5000, class_weight=...)`
- `RandomForestClassifier(n_estimators=500, class_weight=..., random_state=42)`
- `XGBClassifier(objective="multi:softprob", tree_method="hist", learning_rate=0.1, max_depth=6, n_estimators=300, ...)`

...and **the candidate with the best weighted F1 on the held-out test split is kept** as that node's model (`max(results.keys(), key=lambda k: results[k]["f1"])`). All three candidates' scores are preserved in the run summary under `candidate_selection` for transparency even though only the winner gets full ROC/PR artifacts. Example: [`train_domain_model.py`](../models_directory/Classification_Models/Hierarchical_Classification_Model/domain/train_domain_model.py).

At **inference time**, the hierarchical predictor doesn't just use the single "winning" model per node — it actually reloads all three saved per-node artifacts (`lr_domain.pkl`, `rf_domain.pkl`, `xgb_domain.json`, etc.) and, for category/subcategory, takes a **majority vote across the three algorithms** (domain uses XGBoost's prediction directly as final). See [`hierarchical_predictor.py`](../models_directory/Classification_Models/Hierarchical_Classification_Model/hierarchical_predictor.py).

### 6.2 Harm assessment (3 models)
`Harm_Binary`, `Harm_Ordinal_High`, `Harm_Ordinal_Low` — a binary "was there harm at all" gate plus two ordinal sub-models splitting the harm scale further. Embeddings used: `embedding_text123` (harm depends on what was actually done about it, not just the initial complaint).

### 6.3 Severity (1 model, ordinal regression)
Uses `mord.LogisticIT()` — genuine **ordinal logistic regression** (the `mord` package), not a plain multiclass classifier — because severity (Low/Medium/High/[a 4th tier in the raw label space]) is an ordered scale where a Low↔High mistake should be judged worse than a Low↔Medium mistake, which plain multiclass classifiers don't encode. See [`train_severity_model.py`](../models_directory/Classification_Models/Severity_level/train_severity_model.py).

### 6.4 Feedback type & Improvement opportunity type (2 models, ordinal)
Similarly ordinal classifiers over a small ordered label set.

### 6.5 Classification (EN) (1 model)
A flat multiclass classifier over ~76 fine-grained legacy classification codes (the highest-cardinality target in the system, and correspondingly its worst-performing model in current metrics — see §8).

### 6.6 Stage — the one non-ML component
`classify_stage_Score_Based()` is **not a trained classifier** — it's a semantic-similarity **scoring** approach: precomputed centroid embeddings for each stage's vocabulary (`generate_metric_embedding()` embeds a curated word/phrase list per stage and averages the vectors), then a new complaint's embedding (or per-sentence embeddings) is scored against each centroid via cosine similarity (`metric_projection_score`), and the highest-scoring stage wins. This is confirmed as the actual production path (not the alternative `train_ML_Metric_Mapper_Numeric` supervised variant that also exists in the codebase as an experiment). **If replicating this engine, this is a useful pattern for label categories where you have almost no labeled training examples but can hand-curate a vocabulary** — it needs zero training data, just word lists.

### 6.7 Standard evaluation metrics (every model)
`accuracy`, `precision`/`recall`/`f1` (weighted average), a full confusion matrix, and — since Stage 9 of this project — per-class ROC curves + AUC and Precision-Recall curves + Average Precision, with binary vs. one-vs-rest multiclass handled explicitly (`save_evaluation_artifacts()` in `run_versioning.py`). A class with no true instances in a given test split has its curve skipped with a recorded warning rather than crashing or fabricating a number.

---

## 7. Training orchestration & the "Train Now" button

### 7.1 What the user actually clicks
**Settings → Training tab** (`Front_End_Feedback_Analysis/src/components/settings/Training.js`) shows a big **"Train All Models"** button. Clicking it:

1. Calls `POST /api/settings/training/run` ([`training_router.py`](../backend/api/routers/training_router.py)).
2. The backend checks `is_training_running()` first and returns **HTTP 409** if a run is already active — only one training run can be in flight at a time.
3. `run_training_pipeline()` ([`training_service.py`](../backend/api/services/training_service.py)) generates a unique `run_id` (format `YYYY_MM_DD_HHMMSS_<8-char-uuid>`), spins up a **background daemon thread**, and returns immediately with `{"run_id": ..., "status": "started"}` — the HTTP request never blocks on training.
4. The frontend polls `GET /api/settings/training/progress` every **2.5 seconds** while `is_running=true`, showing a live banner: current model name, step X of 18, percentage, and an estimated-remaining-seconds figure computed from the average time-per-step so far.
5. When polling detects the run has stopped, the UI does one more lookup (`GET /api/settings/training/history`) to determine whether it actually **succeeded or failed** — because the progress endpoint alone can't distinguish "finished cleanly" from "crashed 3 seconds in," both look identical (`is_running: false`) from the progress endpoint's point of view.
6. Progress state is written to a **shared JSON file** (`backend/data/training_progress.json`), not just in-memory — this makes it visible across worker processes and survivable across a page refresh, and a startup routine (`reconcile_stuck_training_runs()`) resets a stale `is_running=true` left behind by a killed process, since nothing else would ever clear it otherwise.

### 7.2 Who can click it
Gated by role: `SOFTWARE_ADMIN` or `COMPLAINT_SUPERVISOR` only (`require_role(...)` on every training endpoint) — a regular user cannot see or trigger this tab.

### 7.3 What happens on the backend, step by step
```
run_training_pipeline()
 └─ background thread:
     1. _run_split_data()          # split_data_from_sql_server(): query SQL Server,
                                    # dedupe, 80/20 split, write SQLite hand-off tables,
                                    # record split_metadata.json (method/seed/source/row counts)
     2. _run_train_all(run_id)     # train_all(): runs all 18 trainers IN SEQUENCE
          for each of the 18 models:
            - update_training_progress(model_name, step)   # writes progress file
            - call the trainer function (isolated try/except — one model's
              crash never aborts the other 17; a normalized (None, None)
              contract logs a warning and continues)
            - run_versioning.update_run_summary(run_dir, model_name, metrics)  # atomic write
     3. store_training_run(...)    # persist run + per-model metrics to the
                                    # SQLite-backed training_metadata.db (legacy history log)
     4. record_ml_db_size(...)     # track dataset growth over time for the dashboard chart
```

### 7.4 Where results are written
Every run gets its **own folder**, never overwritten: `models_directory/Classification_Models/Maintainance/training_runs/{run_id}/`, containing:
- `run_summary.json` — full metrics for all 18 models, dataset info (source, row counts, split seed), and which embedding model version produced the training embeddings. Written atomically (temp file + `os.replace`) so it's always valid JSON even if read mid-run.
- Per-model `{model}_confusion_matrix.png`, `{model}_roc_curve.png`, `{model}_pr_curve.png`, `{model}_classification_report.txt`.
- The serialized model artifact itself — `.pkl` (joblib) for sklearn/mord estimators, or native `.json` for XGBoost (XGBoost is never forced through joblib).
- A SHA-256 checksum + size recorded for every artifact in a manifest, which the "Download ZIP" button in the UI (`GET /api/settings/training/versioned-runs/{run_id}/download`) packages up on demand.

This history is also visualized: DB growth over time, F1 trend per model family across the last 10 runs, and a run-duration timeline chart — all served from `training_router.py`'s `/charts/*` endpoints and rendered with **Recharts** in the Training tab.

---

## 8. How long does it actually take?

Measured from real `run_summary.json` files already in this repo (`training_runs/2026_07_20_100634_.../`, `.../2026_07_20_100146_.../`, `.../2026_07_17_164755_.../`), on a dataset of **484 source rows → 387 train / 97 test** after dedup:

| Run | Started | Finished | Duration |
|---|---|---|---|
| `2026_07_20_100634_...` | 10:06:34.44 | 10:07:39.13 | **~65 s** |
| `2026_07_20_100146_...` | 10:01:46.66 | 10:02:54.22 | **~68 s** |
| `2026_07_17_164755_...` | 16:47:55.51 | 16:48:45.88 | **~50 s** |

So a **full retrain of all 18 models currently takes roughly 50–70 seconds** end-to-end (including the SQL Server query + split step) on a dataset this size, on CPU (no GPU is used or required — `torch.device("cuda" if torch.cuda.is_available() else "cpu")` falls back to CPU everywhere, and the embedding model itself is only invoked during the split/embedding-worker phase, not during trainer fitting, since trainers consume already-stored embeddings).

This will **not stay constant as data grows** — sklearn/XGBoost training time scales with row count, and 11 of the 18 models retrain from scratch on the full table every run (no incremental/warm-start training anywhere in this codebase). At low-thousands of rows this is still expected to be well under a few minutes; if porting this design to a dataset with tens of thousands+ of rows, budget time to benchmark this assumption rather than carrying it over blindly.

Separately, **embedding computation** (the background worker, not training) processes in batches of 10 cases every 10-second poll cycle — that cost is amortized continuously as cases are created/edited, not paid in one lump sum at training time, since training reads already-computed embeddings from `ml.CaseTrainingRecord`.

**Cold-start model loading** (the embedding model + all `.pkl`/`.json` inference artifacts) happens once at FastAPI process startup via `startup_ml_warmup()`, which logs its own elapsed time — this is a one-time cost per server restart, not per-request.

---

## 9. Important nuance: training ≠ automatic redeployment

This is the single most important design point to carry into a new project, and it's easy to miss:

> **Clicking "Train All Models" evaluates and versions new candidate models — it does not automatically make them the models used for live predictions.**

Live inference (`classify_feedback()` in [`package_models.py`](../models_directory/Classification_Models/package_models.py)) loads its `.pkl`/`.json` files from **fixed paths** (e.g. `Hierarchical_Classification_Model/domain/vocab_models/lr_domain.pkl`) **once at process import time**. A training run writes its winning models only into `training_runs/{run_id}/...` — a separate, versioned location — specifically so that a bad or unreviewed retrain can never silently swap out what production is using ([`run_versioning.py`](../models_directory/Classification_Models/Maintainance/run_versioning.py), module docstring: *"Training here never writes to a family's fixed live-inference path as a side effect — that would be silent, automatic promotion"*).

Promoting a trained run to production is a **separate, deliberate action** (`promote_run(run_id, family_names)`), and as of this writing it is only implemented (has a known `live_path` target) for **2 of the 18 families** — `harm_binary` and `feedback_type`. It is explicitly **not implemented for `domain`** ("its live-model selection mechanism ... needs investigation before an automated promotion path can be defined") and, in practice, is **not wired to any UI button at all** — it exists as a callable Python function, invoked manually/administratively, not from Settings → Training.

**If building a similar engine from scratch, decide this trade-off up front rather than backing into it:** either (a) keep train/evaluate and promote/deploy as two explicit, separately-authorized steps (safer — matches what this repo does, prevents an accidental bad retrain from immediately going live), or (b) make promotion automatic on every successful run above some quality bar (faster iteration, but requires real quality gates and rollback tooling you'd need to build).

---

## 10. Inference pipeline (how a prediction actually happens)

Entry point: `run_intelligence_pipeline()` in [`prediction_service.py`](../backend/api/services/prediction_service.py), called from the Insert Record page flow.

```
text (raw complaint, e.g. Arabic free text)
  → classify_feedback(text_1=text, text_2="", text_3="")   [package_models.py]
      → get_embedding(text)                     # one embedding model call, 768-dim
      → hierarchical_predict_embeddings(emb)     # domain → category → subcategory cascade
      → predict_severity_from_embedding(emb)     # mord ordinal model
      → classify_stage_Score_Based(text)         # vocabulary-centroid scoring, no embedding model needed here beyond its own precomputed centroids
      → predict_harm_from_embedding(emb)         # binary + 2 ordinals
      → predict_feedback_type_from_embedding(emb)
      → predict_improvement_from_embedding(emb)
      → predict_classification_en_from_embedding(emb)
  → remap_classification_output(...)             # adapts model-space IDs to current live DB IDs
  → returned to the Insert Record UI as suggested field values
```

A `classify_feedback_timed()` variant exists purely for profiling — it wraps every stage in a timer and returns a `timings` dict, useful if you need to know exactly which sub-model is the bottleneck on real hardware.

---

## 11. Technology stack

Pinned versions from [`requirement_py_310_lock.txt`](../requirement_py_310_lock.txt) (Python 3.10):

| Layer | Library | Version |
|---|---|---|
| Web framework | `fastapi` | 0.122.0 |
| ASGI server | `uvicorn` | 0.40.0 |
| DB driver | `pyodbc` (SQL Server) | 5.3.0 |
| ORM/SQL helper | `SQLAlchemy` | 2.0.44 |
| Deep learning | `torch` | 2.9.0 |
| Transformers | `transformers` | 4.57.3 |
| Tokenizers | `tokenizers` | 0.22.1 |
| Classical ML | `scikit-learn` | 1.7.2 |
| Gradient boosting | `xgboost` | 3.1.1 |
| Ordinal regression | `mord` | 0.7 |
| Data handling | `pandas` | 2.3.3, `numpy` 2.2.6 |
| Model serialization | `joblib` | 1.5.2 (+ XGBoost's own native format) |
| Plotting (training artifacts) | `matplotlib` | 3.10.7 |
| Arabic NLP (optional/experimental) | `stanza` | 1.11.0 |
| Speech-to-text (separate subsystem, not covered above) | `faster-whisper` | 1.2.1 |

Frontend (Training tab): **React** + **MUI Joy UI** components, **Recharts** for charts (`AreaChart`/`LineChart`/`BarChart`), plain `fetch`-based API client (`src/api/training.js`).

Database: **Microsoft SQL Server 2022 Express**, single instance, single database (`IncidentManager`) holding both operational (`dbo`) and ML (`ml`) schemas — deliberately not split into two databases or a separate vector store (see §3.2 and the ADR's decision matrix).

No message queue, no Celery/RabbitMQ/Redis anywhere. Both the training pipeline and the embedding worker are **plain Python background threads inside the single FastAPI process**, backed by durable state in SQL tables (job table, progress file) rather than in-memory queues — a deliberate choice for a single-VM, offline hospital deployment where adding infrastructure that needs to stay alive without internet access is a real cost, not a scale requirement (ADR §4.4). **This is very likely the right default for a new project too, unless you already know you need multi-node scale.**

---

## 12. Offline model packaging & distribution

Because this system deploys into air-gapped hospital environments, model artifacts are versioned and pulled as a distinct step, not just baked into `git`:

- [`models.lock.yaml`](../models.lock.yaml) — a lockfile naming a specific **HuggingFace repository + commit revision** (`Abbass-RAH/pfms-feedback-classifier`) as the source of truth for the ~50 trained `.pkl`/`.json` classifier artifacts, with `required_files` explicitly enumerated (so an incomplete download fails loudly, not silently). This does **not** include the 1.1 GB embedding model folder (`model_storage/**` is explicitly excluded) — the embedding model is packaged/distributed separately as part of the offline release bundle, since it doesn't change per training run.
- This two-tier approach (small, frequently-changing classifier heads pulled via a lockfile + pinned revision; large, rarely-changing embedding backbone bundled once) is a reasonable pattern to reuse: **don't re-ship a multi-GB transformer every time a lightweight classifier retrains.**

---

## 13. File map — what to look at first when porting this design

| Concern | File |
|---|---|
| Embedding model load + batch inference | `models_directory/Classification_Models/Stage/modular_functions.py` |
| Background embedding worker | `backend/ml_mapping/embedding_worker.py` |
| Train/test split + SQL Server source query | `models_directory/split_data.py` |
| One worked trainer example (3-candidate selection) | `models_directory/Classification_Models/Hierarchical_Classification_Model/domain/train_domain_model.py` |
| Ordinal regression trainer example | `models_directory/Classification_Models/Severity_level/train_severity_model.py` |
| Orchestrates all 18 trainers | `models_directory/Classification_Models/Maintainance/train_all.py` |
| Versioned run folders, artifacts, promotion | `models_directory/Classification_Models/Maintainance/run_versioning.py` |
| API layer for the Training tab | `backend/api/routers/training_router.py`, `backend/api/services/training_service.py` |
| Frontend "Train All Models" UI | `Front_End_Feedback_Analysis/src/components/settings/Training.js` |
| Live inference entry point | `models_directory/Classification_Models/package_models.py` |
| Hierarchical cascade + majority voting at inference | `models_directory/Classification_Models/Hierarchical_Classification_Model/hierarchical_predictor.py` |
| Offline model artifact packaging | `models.lock.yaml` |

---

## 14. Design lessons worth carrying forward (and pitfalls to avoid)

1. **Put ML data in the same relational database as operational data from day one**, in its own schema, with real foreign keys. Don't split it into a separate SQLite/file store "for now" — this repo's history shows exactly how that drifts (duplicate/orphaned rows, silent failures, no backup story) and how expensive the eventual consolidation was.
2. **Make embedding generation asynchronous and durable** (a status column + job table), never inline/blocking on the request that creates a record, and never a bare `try/except: print()`.
3. **Batch your embedding calls** (`get_embedding_list`) instead of one-at-a-time — this alone was a ~5x reduction in model forward passes here (11 unbatched embeddings/case → 2 batched embeddings/case).
4. **Version every training run into its own immutable folder** with a manifest + checksums, rather than overwriting a single "latest model" location — it's what makes the ROC/PR curves, the download-as-ZIP feature, and safe rollback possible.
5. **Never let a training run silently become the live model.** Keep training/evaluation and promotion/deployment as two distinct, explicit steps until you have real automated quality gates.
6. **Never let one failing model abort the whole batch.** Each of the 18 trainers here is wrapped so one crash just gets logged as a warning and the pipeline continues — critical once you have more than a couple of models in one pipeline.
7. **Record which exact model/version produced a stored embedding.** A mislabeled model name (this repo calls its embedder "MPNet" when it's actually an XLM-RoBERTa checkpoint) is a real, observed failure mode — never trust a name string alone; read the model's own saved config.
8. **A background thread + a durable job/progress table is a legitimate, low-complexity architecture** for a single-instance deployment — you don't need Celery/Redis/RabbitMQ until you actually have a multi-node scale requirement.
9. **For label categories with almost no training data but a well-understood vocabulary, a similarity-to-centroid scoring approach (§6.6) can stand in for a trained classifier** — it needs no labeled examples, just a curated word/phrase list per class.
