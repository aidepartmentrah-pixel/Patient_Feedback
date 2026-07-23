# Offline Installation Dependency Inventory

Every dependency this system needs, where it comes from, and whether it is
resolved **before** the release package is built (online engineering
machine) or **at install/runtime** on the disconnected hospital server. The
air-gap qualification test only proves the system works if every runtime
dependency is already inside the release package — this document is the
checklist behind that claim.

**Principle**: nothing on this list may be fetched over the internet at
install time or at application runtime on the hospital server. Anything
that needs the internet must be resolved once, on an online machine, before
the release package is built (marked **BUILD-TIME (online)** below).

---

## 1. Container base images

| Image | Source | Resolved |
|---|---|---|
| `python:3.11-slim` | Docker Hub | **BUILD-TIME (online)** — pulled when `docker compose build` runs; baked into `backend.tar`/`db-init.tar` |
| `node:20-slim` | Docker Hub | **BUILD-TIME (online)** — build stage only, discarded after `frontend.tar` is built (multi-stage build) |
| `nginx:1.27-alpine` | Docker Hub | **BUILD-TIME (online)** — final frontend runtime image, baked into `frontend.tar` |
| `mcr.microsoft.com/mssql/server:2022-latest` | Microsoft Container Registry | **Not included in this release package** (intentionally — see `RELEASE_NOTES.md` "Release Package Size") — must already be present on the offline server via the separate Offline Debian Server Kit, or pre-pulled and loaded manually before `install_offline.sh` runs |

Verify: `docker images` on the offline server must show all four before
`install_offline.sh` is run — `load_images.sh` handles the first three from
`docker-images/*.tar`; SQL Server is a separate prerequisite.

## 2. Backend Python dependencies

Source: `backend/requirements.txt`, resolved via `pip install` **inside the
Docker build** (`backend/Dockerfile`, `database/sqlserver/Dockerfile`) —
**BUILD-TIME (online)** only. The running container never calls `pip`,
`pypi.org`, or any package index.

Notable packages and why they're there:
- `torch`, `transformers`, `sentencepiece`, `safetensors` — embedding model
  (MPNet/XLM-RoBERTa) used by classification; weights are baked into the
  image via `models_directory/Classification_Models/model_storage/`, not
  downloaded at runtime (see §4).
- `faster-whisper`, `ctranslate2`, `av` — Speech-to-Text; model weights are
  a **separate asset bundle**, not part of `backend.tar` (see §4).
- `xgboost`, `scikit-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn` — the
  classification/training pipeline (`models_directory/`).
- `streamlit`, `pydeck`, `altair` — present in the dependency tree but not
  imported by any backend router/service (unused in production; not part of
  this cleanup's scope, flagged here for a future pruning pass).
- `pyodbc`, `bcrypt`, `cryptography` — SQL Server connectivity and secret
  handling (`SETTINGS_ENCRYPTION_KEY`).
- Removed in this release (see `RELEASE_NOTES.md`): `gliner`, `onnxruntime`,
  `stanza` — were NER-only dependencies, retired along with the feature.

`database/sqlserver/Dockerfile` installs a much smaller, separate set
(`pyodbc`, `bcrypt` only) — it does not need the ML stack at all.

## 3. ML model artifacts (baked into the image, no runtime download)

- `models_directory/Classification_Models/model_storage/mpnet_embeddings/` —
  embedding model weights/tokenizer, `COPY`'d into `backend.tar` at build
  time via `COPY models_directory/ models_directory/` in `backend/Dockerfile`.
- `models_directory/Classification_Models/Hierarchical_Classification_Model/`
  and sibling folders (`Severity_level/`, `Harm_level/`, `Stage/`,
  `Classification_En/`, `feedback_type/`, `improvement_opportunity_type/`) —
  pretrained XGBoost/sklearn model files (`.json`/`.pkl`), same COPY, no
  runtime fetch.
- Each of the 10 category/subcategory model files ships with a small JSON
  label-map sidecar (e.g. `vocab_models/category_domain1_label_map.json`),
  same `COPY`, no runtime fetch, no external database of any kind needed to
  decode a model's output. This replaces an earlier design that queried a
  116MB SQLite file of real patient complaint text
  (`table_feedback_train`) live at import time — see `RELEASE_NOTES.md`
  for the full explanation. Zero patient data travels with this release.
- **Known gap** (see `RELEASE_NOTES.md` and
  `ML_CLASSIFICATION_ISSUE_FOR_DEV_TEAM.md`): Category 1's and Category 2's
  subcategory models are genuinely stale relative to any available
  training data (confirmed on the live production system too, not specific
  to this deployment) and need real retraining. 8 of 9 classification
  outputs work fully offline with real predictions; subcategory for
  categories 1/2 specifically returns `"Not Available"` honestly rather
  than a wrong guess.

## 4. Speech-to-Text model (separate asset, not baked into `backend.tar`)

- Faster-Whisper `medium` model — exported once, on an online engineering
  machine, via `scripts/export_whisper_model.sh`, producing
  `assets/whisper-model-medium.zip` (~1.4GB). `install_offline.sh` extracts
  it locally on the offline server; the backend loads it from disk via the
  `WHISPER_MODEL_PATH` environment variable and never contacts
  huggingface.co. Verify: `assets/whisper-model-medium/` must be non-empty
  before first install (see `VALIDATION_CHECKLIST.md`).

## 5. Frontend dependencies

Source: `Front_End_Feedback_Analysis/package.json`, resolved via `npm ci`
**inside the Docker build's Node stage only** (`Front_End_Feedback_Analysis/Dockerfile`)
— **BUILD-TIME (online)**. The final runtime image is `nginx:1.27-alpine`
serving pre-built static files; no Node.js, no `npm`, and no package
registry access exists in the running container at all.

- Cairo font — self-hosted (`src/assets/fonts/cairo/`), replacing the
  previous `fonts.googleapis.com`/`fonts.gstatic.com` live CDN links. Swept
  the codebase for any remaining references to those domains — none found;
  what remains are code comments documenting the fix, not live calls.
- No other CDN, analytics, or external script/style tags found anywhere in
  `public/index.html` or the built bundle.

## 6. Runtime network calls (what the running system talks to, and why)

| Call | Target | Offline-safe? |
|---|---|---|
| Backend → SQL Server | `sqlserver` container, internal Docker network | Yes — same host/network, no internet |
| Frontend (nginx) → Backend | `/api/*` reverse-proxied to `backend` container | Yes — internal Docker network |
| Backend → Hospital Directory API | Configurable `BaseUrl` (Config → Hospital Directory API) | **Depends on hospital network config** — this is meant to reach an internal hospital system, not the public internet. If the hospital's Hospital Directory API server is itself on an isolated network, confirm reachability from the Docker host as part of deployment, not assumed. |
| Everything else (STT, embeddings, classification models) | Local disk only | Yes — confirmed no huggingface.co/model-hub calls anywhere in `backend/` (see §3, §4) |

## 7. Database schema/data dependencies

- `database/sqlserver/install/002`–`015` — schema, indexes, constraints,
  views, stored procedures, triggers, lookup/config seed data, and the two
  source-ID mapping tables (users, custom views) added in this release. All
  run automatically by `install_database.py` (`sorted(glob("0*.sql"))`
  discovery) inside the `db-init` container — no manual SQL execution
  needed on a fresh install.
- `database/sqlserver/seed/provisioning.v1.json` (+ manifest + checksum) —
  the real organizational units, user accounts, and custom table views
  migrated from the old HCAT system. Supplied via a read-only volume mount
  at runtime (`db-init` service), never baked into any image (see
  `.dockerignore`) — this is the one genuinely install-time-supplied data
  dependency, and `provision.py` fails the install hard if it's missing or
  its checksum doesn't match, rather than silently completing without it.

## 8. What this inventory does NOT cover

- SQL Server's own OS-level dependencies (glibc, ICU, etc.) — internal to
  the `mcr.microsoft.com/mssql/server` image, Microsoft's responsibility,
  not this project's.
- The host Debian server's own package set (Docker Engine, Docker Compose
  plugin) — covered by the separate Offline Debian Server Kit, not this
  application release.

---

**Bottom line**: every dependency this application needs at install time or
runtime is either (a) already inside `docker-images/*.tar`, (b) a separate,
explicitly-verified asset bundle (`assets/whisper-model-medium/`), or (c) a
data artifact shipped alongside the release and checksum-verified before use
(`provisioning.v1.json`). Nothing requires internet access on the
disconnected hospital server, with the one caveat in §6 (Hospital Directory
API reachability is a hospital-network configuration question, not a
software dependency this release can pre-resolve).
