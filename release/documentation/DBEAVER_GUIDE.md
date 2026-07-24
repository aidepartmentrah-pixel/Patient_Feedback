# DBeaver Connection Guide — Patient Feedback System

How to connect DBeaver (or any generic SQL Server client) directly to this
deployment's database for inspection/debugging — separate from the app
itself, which runs on its own ports.

## 1. Port allocation (per the OR-LAB port snapshot)

This app reserves three ports on the OR-LAB host:

| Service | Port | Purpose |
|---|---|---|
| Patient_Feedback Backend | **8100** | REST API — not what DBeaver connects to |
| Patient_Feedback Frontend | **8101** | Web UI — not what DBeaver connects to |
| (reserve, spare) | 8102 | Unused, reserved for later |

**SQL Server itself is a separate port**, not part of that 8100/8101/8102
block — it's whatever `SQLSERVER_HOST_PORT` is set to in this release's
`.env` file (default **1433**, the standard SQL Server port). Confirm the
actual value before connecting:

```bash
grep SQLSERVER_HOST_PORT /opt/pfms-release/.env
```

If it still shows the default, DBeaver connects on `1433`. There is no
conflict with 8100/8101/8102 either way — different port, different service.

## 2. Credentials

From the same `.env` file:

```bash
grep -E "MSSQL_SA_PASSWORD|DB_DATABASE" /opt/pfms-release/.env
```

- **Username**: `sa`
- **Password**: whatever `MSSQL_SA_PASSWORD` is set to (never the
  `__SET_ME__` placeholder — if you see that, `.env` was never filled in
  and the stack won't have started at all)
- **Database**: `IncidentManager` (the `DB_DATABASE` value)

## 3. Create the connection in DBeaver

1. **Database** → **New Database Connection** → choose **SQL Server**
   (Microsoft's driver, not the generic "SQLServer" jTDS variant — DBeaver
   will prompt to download the driver the first time, which requires
   internet access on whatever machine is running DBeaver itself; this is
   independent of the offline server, which never needs internet).
2. **Host**: the OR-LAB server's IP address (not `localhost`, unless
   DBeaver is running on the server itself).
3. **Port**: the `SQLSERVER_HOST_PORT` value from step 1 (`1433` unless
   changed).
4. **Database**: `IncidentManager`.
5. **Authentication**: **SQL Server Authentication** (not Windows
   Authentication — this is a Linux container).
6. **Username** / **Password**: from step 2.
7. Open the **Driver properties** tab and set:
   - `trustServerCertificate` = `true`

   (SQL Server 2022's default self-signed certificate isn't in any trust
   chain DBeaver knows about — without this, the connection fails with a
   certificate validation error, not a credentials error. This mirrors the
   `-C` flag already used everywhere else in this release's own scripts,
   e.g. `verify_installation.sh`'s `sqlcmd` calls.)
8. **Test Connection** → should succeed. **Finish**.

## 4. Firewall note

If the connection times out (not a certificate/credentials error — a
genuine hang), confirm the OR-LAB host's firewall allows inbound traffic on
the SQL Server port from wherever DBeaver is running. The app ports
(8100/8101) being reachable doesn't imply the SQL Server port is too — it's
a separate `docker-compose.yml` port mapping, exposed independently.

## 5. What you'll find once connected

- `dbo.*` — all operational tables (incidents, users, org units, etc.).
- `ml.*` — the ML training-data schema (`CaseTrainingRecord`,
  `HistoricalTrainingExample`, `EmbeddingProcessingJob`, etc.) — see
  `ML_ARCHITECTURE_DECISION_RECORD.md` for what each table is for.

Treat this as a read-mostly inspection tool. Any manual write against a live
production database should go through the application or a reviewed script
(e.g. `database/sqlserver/seed/provision.py`), not ad hoc DBeaver edits —
there's no audit trail for a manual `UPDATE` run by hand.
