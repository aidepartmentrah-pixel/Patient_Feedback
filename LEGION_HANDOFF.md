# Legion Handoff — Database + Model Setup

**Read this file first.** This is the starting brief for setting up a working local
environment on this machine (the Lenovo Legion), as the step before Dockerizing the
Patient_Feedback application. It was written by Claude on the source Windows Server VM
after preparing and *actually testing* everything referenced below — not just generating
it. Follow it in order.

## What this repo is, briefly

`Patient_Feedback` is a hospital complaint-management system: FastAPI backend (SQL
Server), a separate frontend repo (not part of this checkout), and an ML classification
pipeline (scikit-learn/XGBoost models + a shared sentence-transformer embeddings model).
It currently runs in production on a Windows Server VM that cannot run Docker. You're
setting it up here specifically so it can be Dockerized (Prompt 2A in
`4. Application Dockerization & Deployment Playbook.md`, if that file is present /
referenced) — Docker Desktop is available on this machine, it wasn't on the source VM.

Your job right now is **not** to Dockerize yet. It's to get a real, verified, working
database + model set on this machine first, so Dockerization has a known-good target to
containerize rather than guessing.

## The two things you're restoring, and why they're not in git

1. **The database** (`IncidentManager`, SQL Server) — never existed as a portable
   artifact before this session; `database/sqlserver/` is a from-scratch install package
   built by inspecting the live production schema.
2. **The ML models** (`models_directory/Classification_Models/**/*.pkl`, `*.json`, and
   the embeddings model's `*.safetensors`) — deliberately `.gitignore`d (some are large,
   all are binary artifacts, not source). They live in two private Hugging Face repos
   instead and get pulled down with a script.

Both are restored by running scripts, not by hand-copying files.

## Step 1 — Stand up SQL Server via Docker

```bash
docker run -e "ACCEPT_EULA=Y" -e "MSSQL_SA_PASSWORD=<pick-a-strong-password>" \
  -p 1433:1433 --name sqlserver-legion-dev \
  -v sqlserver_legion_data:/var/opt/mssql \
  -d mcr.microsoft.com/mssql/server:2022-latest
```

Wait ~30-60s for it to become healthy (`docker logs sqlserver-legion-dev` until you see
"SQL Server is now ready for client connections").

**Important — the SQL login needs `CREATE DATABASE` permission.** The production login
(`HCAT_Insight`) only has database-level roles scoped to the existing `IncidentManager`
database on the source VM; it does **not** have server-level rights to create a new
database from scratch. On this fresh container, just use `sa` for now — set these
environment variables before running anything below (do not edit
`backend/config/db_settings.json` — env vars override it cleanly, see
`backend/core/config_loader.py`):

```bash
export DB_SERVER=localhost
export DB_DATABASE=IncidentManager
export DB_USERNAME=sa
export DB_PASSWORD=<the password you used above>
export USE_WINDOWS_AUTH=false
export TRUST_SERVER_CERTIFICATE=true
```

(PowerShell: `$env:DB_SERVER = "localhost"`, etc.)

## Step 2 — Run the install pipeline

```bash
python database/sqlserver/scripts/install_database.py
```

This creates the `IncidentManager` database if missing, then runs
`database/sqlserver/install/002` through `011` in order. It's idempotent — safe to
re-run after a partial failure.

**This was tested end-to-end on the source VM against a scratch database before being
handed to you**, and three real bugs were found and fixed in the process (wrong schema
qualifier on some foreign keys, an unsafe implicit datetime conversion, and a driver
quirk with very large single-statement batches). If it still fails for you, that's a new
problem worth investigating carefully, not something to route around — this pipeline is
known to work as of this handoff.

**What you will NOT get, on purpose:** no patient/complaint records, no real user
accounts, no ML training history (`ml.CaseTrainingRecord` / `ml.HistoricalTrainingExample`
will be empty). This is deliberate — see `database/docs/DATABASE_STRUCTURE_REPORT.md`
for the full table-by-table classification and why. Don't try to "fix" this by pulling
that data from anywhere; a fresh install is supposed to look like this.

## Step 3 — Validate the install

```bash
# Run each of these against the new database and compare to the expected notes below
database/sqlserver/validation/validate_tables.sql          # expect: empty result (0 missing, 0 unexpected-obsolete)
database/sqlserver/validation/validate_lookup_data.sql     # expect: every row_count > 0
database/sqlserver/validation/validate_configuration.sql   # expect: 179 / 3 / 0 / 6 rows (OrgUnitPolicy/DepartmentPolicy/DepartmentEvaluationRule/Roles)
database/sqlserver/validation/validate_constraints.sql     # expect: 80 PKs, 55 FKs, 226 non-PK indexes
database/sqlserver/validation/validate_users.sql           # expect: role_definition_count=6, user_account_count=0
database/sqlserver/validation/validate_database_version.sql # expect: exactly 1 row, MigrationName='baseline_install_1.0.0', Success=1
```

These exact numbers were captured from a real successful run on the source VM — if yours
differ, something's actually wrong, don't wave it off.

## Step 4 — Restore the ML models from Hugging Face

You need a Hugging Face access token (ask the user for one, or if they gave you one
already, write it to a file — **never paste it directly into a shell command or commit
it**):

```bash
# from repo root
echo -n "hf_..." > .hf_token
```

Confirm `.hf_token` is listed in `.gitignore` (it should already be — check before
proceeding; if it's missing, add it before doing anything else).

```bash
python scripts/models/download_models.py
```

This reads `models.lock.yaml` (already has both repos and pinned revisions filled in —
`Abbass-RAH/pfms-feedback-classifier` and `Abbass-RAH/shared-mpnet-embeddings`) and
downloads everything into `models_directory/Classification_Models/`. Expect ~172 files
(~173 MB) for the classification models and 6 files (~1.08 GB) for the embeddings model.

## Step 5 — Verify the models

```bash
python scripts/models/verify_models.py
```

Checks SHA-256 against the manifest recorded at upload time, plus a load smoke-test per
file (`joblib.load` for `.pkl`, JSON parse for `.json`, `safetensors.safe_open` for the
embeddings model). Expect all "ok". Some `Stage/Training_Internal_Metrics/vocab_testing/`
JSON files will report a load failure — that's a **known, pre-existing** data-quality gap
(18 of 21 files there are genuinely empty in the source, unrelated to this pipeline, and
not loaded by any inference code). Don't try to fix it; it's flagged, not blocking.

## Guardrails — do not do these

- **Never run anything in `database/sqlserver/retirement/`.** Those scripts `DROP` two
  confirmed-obsolete tables and are deliberately commented out, pending human review —
  not part of setup.
- **Never seed or copy real business/patient data** into this environment from anywhere.
  The whole point of this package is that it doesn't need any.
- **Never commit `.hf_token`.**
- If something about the database or models looks "incomplete" compared to what you'd
  expect from a real production system — that's almost certainly correct, not a bug. Read
  `database/docs/DATABASE_STRUCTURE_REPORT.md` before assuming otherwise.

## When you're done

You should have: a running SQL Server container with a fully-installed, validated
`IncidentManager` schema, and a fully-populated, checksum-verified `models_directory/`.
At that point, tell the user — this is the checkpoint they asked for before starting the
actual Dockerization work (backend Dockerfile, frontend Dockerfile, `docker-compose.yml`,
consuming `database/sqlserver/` for the DB init/migrate service). Don't start
Dockerizing until they confirm.
