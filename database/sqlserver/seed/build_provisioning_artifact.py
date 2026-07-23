"""
Stage B -- deterministic (structurally; NOT byte-deterministic, see below)
transform of Stage A's raw extract into the validated provisioning artifact
that ships in the offline release bundle.

Inputs (database/sqlserver/seed/raw_extract/):
    org_units.json, org_units_manifest.json
    orphaned_units.json          (the 6 structurally-incomplete units)
    user_credentials.json, user_credentials_manifest.json
    custom_views.json, custom_views_manifest.json (the 11 real, currently-
        active Custom Table Views -- see extract_custom_views.py)

Outputs (database/sqlserver/seed/, all gitignored):
    provisioning.v1.json                       -- data only: org_units, users,
                                                   custom_views, flagged_records.
                                                   Bcrypt hashes only, NEVER plaintext.
    provisioning.v1.manifest.json               -- schema_version, source_system,
                                                   timestamps, record_counts,
                                                   checksum_sha256 (external --
                                                   computed over the already-
                                                   written provisioning.v1.json)
    provisioning.v1.json.sha256                 -- plain `sha256sum`-compatible line
    installation_test_credentials.local.json    -- ONE active account per distinct
                                                   role, PLAINTEXT, for the separate
                                                   qualify_offline_installation.sh
                                                   only. Never used by the normal
                                                   installer.

NOTE ON DETERMINISM: bcrypt salts are randomly generated per hash by design.
Running this script twice against the identical raw extract will NOT produce
byte-identical output (different salts -> different PasswordHash strings ->
different provisioning.v1.json bytes -> different checksum). This is correct,
expected behavior, not a bug -- fixed/deterministic salts are never used. The
checksum verifies THIS one already-built artifact wasn't corrupted or tampered
with in transit; it is not a reproducibility guarantee across rebuilds.

Raw plaintext source deletion requires an explicit flag or interactive
confirmation -- never automatic (see the sequence at the bottom of main()).
"""
import argparse
import hashlib
import json
import secrets
import string
import sys
from datetime import datetime, timezone
from pathlib import Path

import bcrypt

SEED_DIR = Path(__file__).resolve().parent
RAW_DIR = SEED_DIR / "raw_extract"
INSTALL_DIR = SEED_DIR.parent / "install"

KNOWN_ROLES = {
    "WORKER", "COMPLAINT_SUPERVISOR", "SECTION_ADMIN",
    "DEPARTMENT_ADMIN", "ADMINISTRATION_ADMIN", "SOFTWARE_ADMIN",
}


def hash_password(password: str) -> str:
    """Identical algorithm to backend/api/db_layer/auth_db.py:hash_password()."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def generate_temp_password(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def load_org_units():
    data = json.load(open(RAW_DIR / "org_units.json", encoding="utf-8"))
    orphaned = json.load(open(RAW_DIR / "orphaned_units.json", encoding="utf-8"))

    units = []
    for a in data["administrations"]["administrations"]:
        units.append({
            "source_id": a["id"], "name": a["name"], "name_ar": a.get("name_ar"),
            "parent_source_id": None, "type": 323, "orphaned": False,
        })
    for d in data["departments"]["departments"]:
        units.append({
            "source_id": d["id"], "name": d["name"], "name_ar": d.get("name_ar"),
            "parent_source_id": d.get("administration_id"), "type": 325, "orphaned": False,
        })
    for s in data["sections"]["sections"]:
        units.append({
            "source_id": s["id"], "name": s["name"], "name_ar": s.get("name_ar"),
            "parent_source_id": s.get("department_id"), "type": 324, "orphaned": False,
        })

    # Structurally orphaned units -- preserved with their REAL name/type as
    # returned by the source, parent_source_id explicitly None (unresolvable,
    # not fabricated), per the approved investigation findings.
    for uid_str, entry in orphaned["units"].items():
        body = entry["body"]
        if body is None:
            continue
        units.append({
            "source_id": body["id"], "name": body["name"], "name_ar": None,
            "parent_source_id": None, "type": body.get("type"), "orphaned": True,
        })

    return units


def load_users():
    users = json.load(open(RAW_DIR / "user_credentials.json", encoding="utf-8"))
    return users


def load_custom_views():
    raw_views = json.load(open(RAW_DIR / "custom_views.json", encoding="utf-8"))

    views = []
    for v in raw_views:
        show_flags = {k: bool(val) for k, val in v.items() if k.startswith("Show")}
        views.append({
            "source_view_id": v["ViewID"],
            "view_name": v["ViewName"],
            "show_flags": show_flags,
            "created_at": v.get("CreatedAt"),
            "source_created_by_user_id": v.get("CreatedByUserID"),
            "is_active": bool(v.get("IsActive", True)),
        })
    return views


def build_flagged_records(org_units, users, unit_ids):
    flagged = []

    for u in org_units:
        if u["orphaned"]:
            flagged.append({
                "kind": "orphaned_org_unit",
                "source_id": u["source_id"],
                "detail": f"Excluded from listing endpoints; empty ancestor chain; "
                          f"name={u['name']!r} type={u['type']!r}. Preserved with "
                          f"parent_source_id=null (not fabricated). See Stage A "
                          f"investigation findings.",
            })
        elif u["name"] is None or str(u["name"]).strip() in ("", "NULL"):
            flagged.append({
                "kind": "null_or_empty_named_unit",
                "source_id": u["source_id"],
                "detail": f"Listed and structurally resolvable, but name={u['name']!r}. "
                          f"type={u['type']!r}.",
            })

    for u in users:
        reasons = []
        if not u.get("role"):
            reasons.append("no role assigned")
        if not u.get("org_unit_id"):
            reasons.append("no org_unit_id")
        elif u["org_unit_id"] not in unit_ids:
            reasons.append(f"org_unit_id={u['org_unit_id']} does not resolve to any known unit")
        haystack = f"{u['username']} {u.get('org_unit') or ''}".lower()
        if any(k in haystack for k in ("test", "what so ever", "universal_section_user", "whatsoever")):
            reasons.append("test-looking username or org unit name")
        if reasons:
            flagged.append({
                "kind": "flagged_user_account",
                "username": u["username"],
                "source_user_id": u["user_id"],
                "active": u.get("active"),
                "reasons": reasons,
            })

    return flagged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete-sensitive-source", action="store_true",
                         help="Delete raw_extract/user_credentials.json after a "
                              "successful, validated build, without prompting.")
    args = parser.parse_args()

    print("=== Stage B: building provisioning artifact ===\n")

    org_units = load_org_units()
    users = load_users()
    custom_views = load_custom_views()
    unit_ids = {u["source_id"] for u in org_units}

    print(f"Loaded {len(org_units)} org units ({sum(1 for u in org_units if u['orphaned'])} orphaned)")
    print(f"Loaded {len(users)} users")
    print(f"Loaded {len(custom_views)} custom views")

    provisioned_users = []
    handoff_temp_passwords = []  # accounts with no recoverable password (none expected this run)
    for u in users:
        test_password = u.get("test_password")
        if test_password:
            password_hash = hash_password(test_password)
            provenance = "migrated_temp_hash"
        else:
            fresh = generate_temp_password()
            password_hash = hash_password(fresh)
            provenance = "freshly_generated"
            handoff_temp_passwords.append({
                "username": u["username"], "role": u.get("role"), "password": fresh,
            })
        provisioned_users.append({
            "source_user_id": u["user_id"],
            "username": u["username"],
            "display_name": u.get("display_name"),
            "role": u.get("role"),
            "org_unit_source_id": u.get("org_unit_id"),
            "org_unit_type": u.get("org_unit_type"),
            "active": bool(u.get("active")),
            "password_hash": password_hash,
            "password_provenance": provenance,
        })

    flagged_records = build_flagged_records(org_units, users, unit_ids)

    provisioning_data = {
        "org_units": org_units,
        "users": provisioned_users,
        "custom_views": custom_views,
        "flagged_records": flagged_records,
    }

    # ---- Write atomically (temp file + rename) ----
    final_path = SEED_DIR / "provisioning.v1.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(provisioning_data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(final_path)
    print(f"\nWrote {final_path}")

    # ---- Validate the artifact ----
    reloaded = json.load(open(final_path, encoding="utf-8"))
    assert len(reloaded["org_units"]) == len(org_units)
    assert len(reloaded["users"]) == len(provisioned_users)
    assert len(reloaded["custom_views"]) == len(custom_views)
    assert all(v["view_name"] for v in reloaded["custom_views"]), "every custom view must have a name"
    role_codes_seen = {u["role"] for u in reloaded["users"] if u["role"]}
    unknown_roles = role_codes_seen - KNOWN_ROLES
    if unknown_roles:
        print(f"ERROR: unknown role codes found: {unknown_roles}")
        sys.exit(1)
    print("Validation: structure OK, all role codes recognized.")

    # ---- Verify every user got a valid bcrypt hash (spot-check round-trip) ----
    sample_checks = 0
    for u in provisioned_users[:10] + provisioned_users[-10:]:
        raw = next(x for x in users if x["user_id"] == u["source_user_id"])
        check_pw = raw.get("test_password") or next(
            h["password"] for h in handoff_temp_passwords if h["username"] == u["username"]
        )
        if not bcrypt.checkpw(check_pw.encode("utf-8"), u["password_hash"].encode("utf-8")):
            print(f"ERROR: bcrypt round-trip failed for {u['username']}")
            sys.exit(1)
        sample_checks += 1
    print(f"Validation: bcrypt round-trip verified on {sample_checks} sampled accounts.")

    # ---- External manifest + checksum (computed over the FINAL file's bytes) ----
    file_bytes = final_path.read_bytes()
    checksum = hashlib.sha256(file_bytes).hexdigest()

    role_counts = {}
    for u in provisioned_users:
        role_counts[u["role"] or "(none)"] = role_counts.get(u["role"] or "(none)", 0) + 1

    manifest = {
        "schema_version": 1,
        "source_system": "HCAT-170.70.32.34",
        "extracted_at": json.load(open(RAW_DIR / "user_credentials_manifest.json", encoding="utf-8"))["extracted_at"],
        "transformed_at": datetime.now(timezone.utc).isoformat(),
        "checksum_sha256": checksum,
        "record_counts": {
            "org_units_total": len(org_units),
            "org_units_orphaned": sum(1 for u in org_units if u["orphaned"]),
            "users_total": len(provisioned_users),
            "users_active": sum(1 for u in provisioned_users if u["active"]),
            "users_inactive": sum(1 for u in provisioned_users if not u["active"]),
            "users_by_role": role_counts,
            "users_freshly_generated_password": len(handoff_temp_passwords),
            "custom_views_total": len(custom_views),
            "flagged_records": len(flagged_records),
        },
    }
    manifest_path = SEED_DIR / "provisioning.v1.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {manifest_path}")

    sha256_path = SEED_DIR / "provisioning.v1.json.sha256"
    with open(sha256_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{checksum}  provisioning.v1.json\n")
    print(f"Wrote {sha256_path}")

    # ---- Qualification test credentials: one ACTIVE account per distinct role ----
    qual_accounts = []
    seen_roles = set()
    for u in users:
        role = u.get("role")
        if role and u.get("active") and role not in seen_roles and u.get("test_password"):
            qual_accounts.append({"role": role, "username": u["username"], "password": u["test_password"]})
            seen_roles.add(role)
    qual_path = SEED_DIR / "installation_test_credentials.local.json"
    with open(qual_path, "w", encoding="utf-8") as f:
        json.dump({"accounts": qual_accounts}, f, ensure_ascii=False, indent=2)
    print(f"Wrote {qual_path} ({len(qual_accounts)} representative accounts, one per role)")

    if handoff_temp_passwords:
        handoff_path = SEED_DIR / "fresh_password_handoff.local.json"
        with open(handoff_path, "w", encoding="utf-8") as f:
            json.dump({"accounts": handoff_temp_passwords}, f, ensure_ascii=False, indent=2)
        print(f"Wrote {handoff_path} ({len(handoff_temp_passwords)} accounts needing password handoff to hospital IT)")

    # ---- Transformation report ----
    print("\n=== Transformation report ===")
    print(json.dumps(manifest["record_counts"], indent=2, ensure_ascii=False))
    print(f"\nAccounts with a freshly generated password (no recoverable source password): {len(handoff_temp_passwords)}")
    print(f"Flagged records: {len(flagged_records)}")

    # ---- Raw source deletion: requires explicit flag or interactive confirmation ----
    raw_users_path = RAW_DIR / "user_credentials.json"
    if args.delete_sensitive_source:
        raw_users_path.unlink(missing_ok=True)
        print(f"\n--delete-sensitive-source given: deleted {raw_users_path}")
    else:
        print(f"\n{raw_users_path} still contains plaintext passwords.")
        answer = input("Delete it now that the artifact is validated? [y/N]: ").strip().lower()
        if answer == "y":
            raw_users_path.unlink(missing_ok=True)
            print("Deleted.")
        else:
            print("Left in place -- delete manually when ready.")


if __name__ == "__main__":
    main()
