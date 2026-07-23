# Troubleshooting Guide — Patient Feedback System

## Backend stays "health: starting" for a long time, or fails at STT warmup

**Most likely cause:** the Speech-to-Text model asset wasn't extracted, or
extracted somewhere the backend can't see. Check:

```bash
./scripts/show_logs.sh backend
```

A line like:

```
[STT WARMUP] Failed after ... : WHISPER_MODEL_PATH='/models/whisper-medium' does not exist or is not a directory.
```

means `assets/whisper-model-medium/` is missing or empty. Confirm it was
extracted:

```bash
ls -la assets/whisper-model-medium/
```

Expected: several files (`model.bin`, `config.json`, `tokenizer.json`,
etc.), not an empty directory. If empty or missing, re-run:

```bash
unzip -o assets/whisper-model-medium.zip -d assets/
```

then restart the backend: `docker restart pfms-backend`.

This should NOT require any Internet access — the model ships as
`assets/whisper-model-medium.zip` inside this release package and is
extracted locally. If you see the backend trying to reach `huggingface.co`
in the logs at all, something is misconfigured (e.g. `WHISPER_MODEL_PATH`
not set) — check the `backend` service's `environment:` block in
`compose/docker-compose.yml`.

## `Permission denied` errors in backend logs (`.cache/matplotlib`)

Harmless — matplotlib falls back to a temp directory automatically and
this does not affect functionality. Not related to the Speech-to-Text
model, which is a plain read-only bind mount, not a Docker-managed volume,
so it does not have this class of ownership issue.

## `db-init` container shows a non-zero exit code

```bash
./scripts/show_logs.sh db-init
```

Look for a line like `<filename>.sql FAILED` followed by the SQL error.
Do **not** re-run `install_offline.sh` repeatedly hoping it fixes itself —
the install scripts are idempotent but a genuine schema/data problem will
fail the same way every time. Escalate with the exact error text.

## `db-init` fails with "provisioning.v1.json not found" or a checksum error

The organizational-unit/user provisioning artifact is missing or corrupted.
This is caught twice — once by `install_offline.sh` itself before the stack
even starts, and again inside `db-init` as defense-in-depth. Either way, the
release package is incomplete or was corrupted in transit:

```bash
ls -la database/sqlserver/seed/provisioning.v1.json*
cd database/sqlserver/seed && sha256sum -c provisioning.v1.json.sha256
```

If the checksum fails, re-copy the entire release package from its source —
do not try to "repair" the file. Do not proceed with a mismatched checksum.

## `db-init` fails with "DRIFT DETECTED"

`provision.py` found a row already in the database that **differs** from
what the release's `provisioning.v1.json` expects (e.g. an organizational
unit's name changed, or a user's role/scope changed) — this is a deliberate
fail-hard, not a bug. The exact log line names the table, the record, and
both the existing and incoming values:

```
=== DRIFT DETECTED -- provisioning FAILED, rolled back entirely ===
AdminsrationUnit 5 drift: existing={...} incoming={...}
```

The whole provisioning step rolls back — nothing partial is left committed.
This will happen if you're re-running `install_offline.sh`/`update_offline.sh`
against a database that already has *manually edited* organizational data
that no longer matches the release. Investigate which value is correct
before doing anything — do not just delete the conflicting row to make the
error go away.

Note: `APP_Users.PasswordHash` is the one exception — it is **never**
overwritten on rerun even if it differs (e.g. someone already reset a
migrated account's password locally), and this does not cause a drift
failure.

## Backend can't connect to the database (`"connected":false` from `/api/status`)

```bash
curl http://localhost:8100/api/status
```

Check:
1. Is `sqlserver` healthy? `docker compose ps` should show it `healthy`.
2. Does `.env`'s `MSSQL_SA_PASSWORD` match what SQL Server was actually
   started with? (Changing this in `.env` after first install does **not**
   change the already-running SQL Server's password.)
3. Check backend logs for the specific ODBC error:
   `./scripts/show_logs.sh backend`

## Frontend loads but shows API errors / blank data

Test the reverse proxy directly:

```bash
curl -i http://localhost:8101/api/status
```

Expected: `HTTP/1.1 200 OK` with a JSON body. If you get a different
status or a connection error, check nginx's config is present and backend
is reachable from the frontend container:

```bash
docker exec pfms-frontend wget -qO- http://backend:8100/api/status
```

## A port is already in use when starting the stack

```bash
ss -tulpn | grep <port-number>
```

This shows what's already using it. Either stop that process/container, or
change the conflicting port in `.env` (`SQLSERVER_HOST_PORT`,
`BACKEND_HOST_PORT`, `FRONTEND_HOST_PORT`) and re-run
`./scripts/start_stack.sh`.

## Disk space running low

```bash
df -h
docker system df
```

To reclaim space from old, unused images (only after confirming you don't
need them for rollback):

```bash
docker image prune -a
```

**Do not** run `docker system prune --volumes` — this can delete the
database volume. Never run bulk volume-pruning commands on this server.

## Something looks wrong after a restart, but was fine before

```bash
./scripts/verify_installation.sh
```

This runs the same checks used right after install/update and will
pinpoint which layer (containers, backend API, frontend, or database) is
the actual problem.

## Still stuck

Collect these before escalating:

```bash
docker compose --env-file .env -f compose/docker-compose.yml ps
./scripts/show_logs.sh > /tmp/pfms-all-logs.txt   # Ctrl+C after a few seconds
./scripts/verify_installation.sh
```
