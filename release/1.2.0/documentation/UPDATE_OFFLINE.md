# Update Guide — Patient Feedback System

Use this procedure to move an already-installed system to a newer release.

## Before you start

- Obtain the new release package (a folder with a newer `docker-images/`,
  same structure as this one).
- Confirm the new release's `RELEASE_NOTES.md` for anything version-specific
  (schema changes, new required `.env` values, etc.) before proceeding.
- Do not delete the currently-installed release folder — you may need its
  images to roll back.

## Step 1 — Copy the new images in

Copy the new release's `docker-images/*.tar` files into your **currently
installed** release folder's `docker-images/` directory, replacing the old
ones. Update `APP_VERSION` in `.env` to the new version.

## Step 2 — Run the updater

```bash
cd /opt/pfms-release
./scripts/update_offline.sh
```

This will, in order:

1. **Back up the database** (via `backup_database.sh`) — note the printed
   backup filename, you'll need it if you have to roll back.
2. **Load the new images.**
3. **Recreate** the `db-init`, `backend`, and `frontend` containers. The
   `sqlserver` container and its data are **not** touched.
4. **Run verification** (`verify_installation.sh`) automatically.

Expected final output: `=== Update complete ===`.

## If verification fails after updating

The script prints rollback steps. In short:

```bash
# 1. Put the previous release's image .tar files back and load them:
docker load -i docker-images/backend.tar     # (the OLD version's file)
docker load -i docker-images/frontend.tar
docker load -i docker-images/db-init.tar

# 2. Restore the pre-update backup (filename was printed in step 1 above):
./scripts/restore_database.sh <the_backup_filename.bak>

# 3. Recreate containers on the old images:
docker compose --env-file .env -f compose/docker-compose.yml -p pfms up -d --force-recreate
```

Then run `./scripts/verify_installation.sh` again to confirm the rollback
succeeded, and investigate the failure before attempting the update again.

## Note on database migrations

This release (1.0.0) is the baseline install — there is nothing to migrate
yet. `update_offline.sh` re-runs the same install scripts used for a fresh
install; this is safe because every script uses `IF NOT EXISTS` /
`IF OBJECT_ID IS NULL` guards and will not touch existing data. A future
release that changes the database schema will ship real files under
`database/sqlserver/migrations/` and this guide (and `update_offline.sh`)
should be revised to apply them explicitly, in order, rather than re-running
the baseline install.
