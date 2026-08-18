# Bug: `provisioning.v1.json.sha256` is stale in the packaged release — every fresh install of PFMS 1.0.0 fails at Step 4/6

**Found during:** offline install validation of Patient Feedback System
release `1.0.0`, on the RAH-OIP lab's Offline Validation VM, 2026-07-24.

**Severity:** Blocks every fresh install. `install_offline.sh` treats a
checksum mismatch here as a hard-stop by design ("do not proceed with a
re-copy... investigate"), so this is not something an operator can safely
work around without editing files inside the release package.

## Symptom

```
[4/6] Verifying the organizational-unit/user provisioning artifact ...
sha256sum: WARNING: 1 computed checksum did NOT match
ERROR: provisioning.v1.json failed checksum verification.
```

## Root cause

`database/sqlserver/seed/provisioning.v1.json.sha256` and the
`checksum_sha256` field in `provisioning.v1.manifest.json` both record
`af8a5834ebb519de8579ea43d7f3bb0d9059a32bbea889f488ad1c6e9dc72eaa`.

The actual, current `provisioning.v1.json` in the shipped release hashes to
`596dc133581012a2181d11dfef84430655212c7ae289f471d5026d8acd711a9c` — this is
also the value recorded for that file in the release's own top-level
`checksums/release_hashes.txt`, so the *package* manifest and the *per-file*
checksum disagree with each other.

`merge_custom_views_into_artifact.py` does correctly recompute and rewrite
both `provisioning.v1.json.sha256` and the manifest's `checksum_sha256`
after it merges `custom_views` in (confirmed by reading the script — it
hashes the final file bytes and writes both). So the stale value predates
whatever step produced the copy that shipped in this release — i.e.
`provisioning.v1.json` was modified again *after* the merge script's
checksum write, without re-running it (or its equivalent step) a second
time before packaging.

**This is not a transfer/corruption issue** — verified by comparing
`sha256sum provisioning.v1.json` against `provisioning.v1.json.sha256` on
two independent copies (the source copy still on the Legion engineering
workstation, and a freshly re-copied, checksum-verified-against-
`release_hashes.txt` copy on the offline test VM) — both show the same
mismatch, so it's baked into the release as built, not something that
happened in transit.

## Content is valid, just unattested

The current `provisioning.v1.json` content matches what the manifest's
`record_counts` describes (179 org units, 162 users, 11 custom_views) — this
looks like the *correct*, current artifact, just with a stale checksum
sitting next to it. Worth double-checking there isn't a second, truly-older
copy of `provisioning.v1.json` that actually matches
`af8a5834...` sitting somewhere in the build pipeline that should have
shipped instead — but on its face the shipped `.json` looks right and the
`.sha256`/manifest look wrong.

## Suggested fix

Whatever packaging/release step runs after
`merge_custom_views_into_artifact.py` (or any other later edit to
`provisioning.v1.json`) needs to either (a) not touch the file again after
checksums are written, or (b) recompute
`provisioning.v1.json.sha256` + the manifest's `checksum_sha256` as its
last action before the release is zipped/copied out. Consider adding an
automated release-build check that re-verifies every per-file `.sha256`
next to `release_hashes.txt` generation, so this class of bug fails the
build instead of failing at every customer's install.

## Workaround used for this validation run

Regenerated the checksum from the actual shipped file content on our test
copy only (not the source on the Legion):

```bash
cd database/sqlserver/seed
sha256sum provisioning.v1.json > /tmp/x
# rewrote provisioning.v1.json.sha256 and manifest's checksum_sha256 to match
```

Install then proceeded normally. **This workaround was not applied to the
release package on the Legion (`C:\Users\it\Documents\GitHub\Patient_Feedback\release`)
— that copy still has the bug and will fail the same way for anyone else
who installs from it as-is**, so this still needs a real fix upstream
before this release goes to a real hospital server.
