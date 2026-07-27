# Linux Commands Reference — Patient Feedback System

Written for an operator with limited Linux experience. Every command below
is exact — copy/paste it, replacing anything in `<angle brackets>`.

## Navigating to the release folder

```bash
cd /opt/pfms-release
```

## Checking container status

```bash
docker compose --env-file .env -f compose/docker-compose.yml -p pfms ps
```

Expected columns: `NAME`, `STATUS` (look for `healthy`, `Up`, or for
`db-init`, `Exited (0)` which is normal), `PORTS`.

## Viewing logs

```bash
./scripts/show_logs.sh              # all services
./scripts/show_logs.sh backend      # just the backend
./scripts/show_logs.sh frontend
./scripts/show_logs.sh sqlserver
```

Press `Ctrl+C` to stop following logs (this does not stop the containers).

## Starting / stopping

```bash
./scripts/start_stack.sh
./scripts/stop_stack.sh
```

## Restarting a single container

```bash
docker restart pfms-backend
docker restart pfms-frontend
```

## Checking disk space

```bash
df -h
```

Look at the `Use%` column for the filesystem containing `/var/lib/docker`
(usually `/`). Above 85% warrants attention.

## Checking memory

```bash
free -h
```

## Checking what's using a port

```bash
ss -tulpn | grep <port-number>
# example:
ss -tulpn | grep 8100
```

## Checking Docker itself is running

```bash
sudo systemctl status docker
```

Expected: `Active: active (running)` in green.

## Getting a shell inside a container (for advanced troubleshooting)

```bash
docker exec -it pfms-backend sh
docker exec -it pfms-sqlserver bash
```

Type `exit` to leave.

## Copying a file out of a container

```bash
docker cp pfms-sqlserver:/var/opt/mssql/backup/<filename>.bak ./
```

(Not usually needed — `backups/` is already bind-mounted to the host, see
`BACKUP_RESTORE.md`.)

## Checking free space inside a Docker volume

```bash
docker system df -v
```
