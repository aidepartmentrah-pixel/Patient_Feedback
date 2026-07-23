# Portainer Deployment Guide — Patient Feedback System

This is an alternative to running the `scripts/*.sh` files by hand: managing
this stack through Portainer's web interface. Portainer itself ships with
the Offline Debian Server Kit, not with this release.

## 1. Open Portainer

In a browser, go to:

```
https://<server-ip>:9443
```

(Port may differ — check the Offline Debian Server Kit's documentation for
this server's actual Portainer port, e.g. see the OR-LAB port snapshot,
where Portainer is on 8000/9443.)

## 2. Load the images first

Portainer does not load `.tar` files for you. Before creating the stack,
load the images from the command line once:

```bash
cd /opt/pfms-release
./scripts/load_images.sh
```

Then in Portainer, go to **Images** in the left sidebar and confirm you see:

- `rah-pfms-backend:1.0.0`
- `rah-pfms-frontend:1.0.0`
- `rah-pfms-db-init:1.0.0`
- `mcr.microsoft.com/mssql/server:2022-latest`

Also extract the Speech-to-Text model asset (Portainer does not do this for
you either):

```bash
unzip -o assets/whisper-model-medium.zip -d assets/
```

## 3. Create the stack

1. In Portainer, go to **Stacks** → **Add stack**.
2. Name it `pfms` (or match `PROJECT_NAME` in your `.env`).
3. Choose **Upload** and select
   `/opt/pfms-release/compose/docker-compose.yml`, or paste its contents
   into the web editor.
4. Under **Environment variables**, add every value from your filled-in
   `.env` file (or use Portainer's "Load variables from .env file" upload
   option, pointing at `/opt/pfms-release/.env`).
5. Click **Deploy the stack**.

## 4. Verify

Go to **Containers** and confirm all four show a green/healthy status:
`pfms-sqlserver`, `pfms-backend`, `pfms-frontend`, plus `db-init` which
should show **Exited (0)** — that's expected, it's a one-shot job, not a
long-running service.

If `backend` shows "starting" for more than ~2 minutes, check the container
logs — see `TROUBLESHOOTING.md` for the Speech-to-Text model asset check.

## 5. View logs

Click a container name → **Logs** tab. Or from the command line:

```bash
./scripts/show_logs.sh backend
```

## 6. Restart / stop services

In Portainer: select the container(s) → **Restart** or **Stop** from the
toolbar above the container list. From the command line:

```bash
./scripts/stop_stack.sh
./scripts/start_stack.sh
```

## 7. Confirm SQL Server is running

**Containers** → `pfms-sqlserver` should show **running / healthy**. To
confirm the database itself from the command line:

```bash
./scripts/verify_installation.sh
```

## Updating the stack in Portainer

1. Load the new images (`./scripts/load_images.sh` after copying the new
   `docker-images/*.tar` files in, as described in `UPDATE_OFFLINE.md`).
2. In Portainer, open the `pfms` stack → **Editor** → update the
   `APP_VERSION` environment variable → **Update the stack**, with
   **"Re-pull image and redeploy"** left unchecked (there is no registry to
   pull from — Portainer will use the already-loaded local image matching
   the tag).

Prefer running `./scripts/update_offline.sh` from the command line instead
when possible — it performs the mandatory pre-update database backup
automatically, which the Portainer UI does not do for you.
