# Backup and Restore Guide — Patient Feedback System

## Where backups are stored

`scripts/backup_database.sh` writes SQL Server `.bak` files to the
`backups/` folder next to this documentation folder (i.e.
`<release-root>/backups/`), **on the host**, not just inside the container
— you can copy these off the server (USB, network share, etc.) using normal
Linux file tools.

## Creating a backup

```bash
cd /opt/pfms-release
./scripts/backup_database.sh
```

Expected output:

```
=== Backing up IncidentManager ===
...
=== Backup complete ===
File: /opt/pfms-release/backups/IncidentManager_20260722_143000.bak
```

**Always back up before:**
- Running `update_offline.sh` (it does this automatically).
- Any manual database change.
- Any infrastructure maintenance affecting the SQL Server container.

## Restoring a backup

```bash
cd /opt/pfms-release
./scripts/restore_database.sh IncidentManager_20260722_143000.bak
```

This is destructive — it **replaces** all current data in the database. The
script asks you to type `YES` to confirm before doing anything. It stops
the backend first (so it isn't holding open connections during the
restore), performs the restore, then restarts the backend.

Run `./scripts/verify_installation.sh` afterward to confirm the system is
healthy again.

## Listing available backups

```bash
ls -lh /opt/pfms-release/backups/
```

Or simply run `restore_database.sh` with no arguments — it lists them for
you.

## Copying a backup off the server

```bash
# To a USB drive mounted at /media/usb:
cp /opt/pfms-release/backups/IncidentManager_20260722_143000.bak /media/usb/
```

Consult the hospital's data-handling policy for where backups may be
stored — this file contains real patient/administrative data once the
system has been in use.
