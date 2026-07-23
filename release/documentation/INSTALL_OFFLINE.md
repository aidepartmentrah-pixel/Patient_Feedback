# Offline Installation Guide — Patient Feedback System

Audience: an IT operator with limited Linux/Docker experience, installing
this application on a Debian offline server for the first time.

## Prerequisites

Before you start, confirm on the target server:

```bash
docker --version
docker compose version
```

Expected: both commands print a version number. If either fails, install
Docker first from the Offline Debian Server Kit (`OFFLINE_SERVER_INSTALL_MANUAL.md`
in that kit) — this release does not install Docker itself.

Also confirm the SQL Server image is already present (it ships with the
Server Kit, not with this release):

```bash
docker image inspect mcr.microsoft.com/mssql/server:2022-latest
```

Expected: a block of JSON (image details). If you instead see
`Error: No such image`, load it from the Server Kit before continuing.

## Step 1 — Copy the release to the server

Copy the entire release folder (the one containing this file, `scripts/`,
`compose/`, etc.) to a location on the offline server, for example:

```bash
/opt/pfms-release/
```

## Step 2 — Configure the environment

```bash
cd /opt/pfms-release
cp .env.offline.template .env
nano .env
```

Fill in every value marked `__SET_ME__`:

- `APP_VERSION` — must match the version tag on the images you loaded
  (check `docker-images/` filenames, e.g. `1.0.0`).
- `MSSQL_SA_PASSWORD` — pick a strong password. Record it in the hospital's
  approved credential store.
- `MSSQL_PID` — the SQL Server edition your license covers
  (`Express`, `Standard`, `Enterprise`, `EnterpriseCore`). **Do not use
  `Developer` in production** — it is not licensed for that.
- `SETTINGS_ENCRYPTION_KEY` — required for the Hospital Directory API
  settings page to work. Generate a real one after Step 3 loads the backend
  image:
  ```bash
  docker run --rm rah-pfms-backend:1.0.0 python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
  ```
  Record it in the hospital's approved credential store — losing it makes
  any previously-saved API key undecryptable (the app reports this clearly
  and lets you re-save, it does not crash).

Save and exit (in `nano`: Ctrl+O, Enter, Ctrl+X).

## Step 3 — Run the installer

```bash
cd /opt/pfms-release
./scripts/install_offline.sh
```

This will:

1. Verify Docker is present.
2. Load the three application images from `docker-images/*.tar`.
3. Extract the Speech-to-Text model asset (`assets/whisper-model-medium.zip`).
4. Verify `database/sqlserver/seed/provisioning.v1.json`'s checksum (fails
   immediately, before starting anything, if missing or corrupted).
5. Start SQL Server, wait for it to be healthy, then run the database
   installer automatically — this installs the schema **and** provisions
   real organizational units and user accounts (179 units, 162 accounts) in
   one automatic step. A fresh install can no longer complete with an empty
   organization/accounts table.
6. Start the backend and frontend, and wait for the backend to report
   healthy.

If step 5's provisioning detects the database already has *different* data
than the release expects for an organizational unit or user scope (drift),
it fails the whole installation with a specific report rather than silently
overwriting or ignoring the difference — see `TROUBLESHOOTING.md`.

Expected final output:

```
=== Installation complete. Backend is healthy. ===

Application URL:  http://<server-ip>:8101
Backend API docs: http://<server-ip>:8100/docs
```

If the script exits with a warning about the backend not becoming healthy
within the wait window, check `./scripts/show_logs.sh backend` — see
`TROUBLESHOOTING.md`.

## Step 4 — Validate

```bash
./scripts/verify_installation.sh
```

Expected: every line reads `[PASS]`, ending with
`Summary: N passed, 0 failed`.

## Step 5 — Confirm in a browser

From a machine on the same network as the server, open:

```
http://<server-ip>:8101
```

You should see the Patient Feedback System login page.

## What to do if something fails

See `TROUBLESHOOTING.md`.
