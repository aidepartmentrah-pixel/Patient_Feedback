# Patient Feedback System — Offline Release Package

**Start here.** This folder is a complete, self-contained offline release.
Copy this entire folder to the target Debian server and follow
`documentation/INSTALL_OFFLINE.md`.

## Folder contents

```
release/
├── README.md                    <- you are here
├── .env.offline.template        <- copy to .env and fill in before install
├── docker-images/                <- backend.tar, frontend.tar, db-init.tar
├── compose/
│   └── docker-compose.yml       <- production compose file (images only, no build)
├── database/sqlserver/           <- install/migration/validation/rollback SQL, for reference
├── scripts/                      <- install_offline.sh, update_offline.sh, backup/restore, etc.
├── documentation/                <- operator guides (start with INSTALL_OFFLINE.md)
└── checksums/                    <- release_hashes.txt (integrity verification)
```

## Read these in order

1. `documentation/RELEASE_NOTES.md` — what's in this release and its
   known gaps (read this first, especially the Speech-to-Text model note).
2. `documentation/INSTALL_OFFLINE.md` — first-time installation.
3. `documentation/VALIDATION_CHECKLIST.md` — sign-off checklist after
   installing.
4. `documentation/UPDATE_OFFLINE.md` — for future updates.
5. `documentation/BACKUP_RESTORE.md`, `TROUBLESHOOTING.md`,
   `LINUX_COMMANDS_REFERENCE.md`, `PORTAINER_GUIDE.md` — as needed.

## Verifying this package wasn't corrupted in transit

```bash
cd release
sha256sum -c checksums/release_hashes.txt
```

Every line should read `OK`. If any line reads `FAILED`, re-copy this
release package before proceeding.
