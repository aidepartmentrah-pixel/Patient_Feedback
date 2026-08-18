"""
Stage C -- provisions real organizational units and user accounts/scopes from
the validated provisioning artifact (provisioning.v1.json) into the target
SQL Server database. Runs inside a single transaction; rolls back completely
on any error. Idempotent: safe to re-run, with an explicit per-table drift
policy (see DRIFT POLICY below) rather than blindly overwriting on rerun.

Usage:
    python database/sqlserver/scripts/../seed/provision.py [--dry-run]

Connection comes from backend/config/db_settings.json (+ env var overrides),
same as install_database.py.

DRIFT POLICY (see the approved migration plan):
    AdminsrationUnit       -- identical -> skip; different -> FAIL (whole run)
    APP_Users.PasswordHash -- never overwritten on rerun, ever
    APP_Users (other cols) -- reconciled (updated to match the artifact)
    APP_UserRoleScope      -- identical -> skip; different -> FAIL (whole run)

No table's rerun path reports success while silently ignoring a real mismatch.
"""
import argparse
import base64
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
BACKEND_DIR = REPO_ROOT / "backend"
SEED_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(BACKEND_DIR))
from core.deployment_port import (  # noqa: E402
    DB_SERVER, DB_DATABASE, DB_DRIVER,
    USE_WINDOWS_AUTH, DB_USERNAME, DB_PASSWORD,
    TRUST_SERVER_CERTIFICATE,
)
import pyodbc  # noqa: E402

SOURCE_SYSTEM = "HCAT-170.70.32.34"


def _conn_string(database: str) -> str:
    parts = [f"DRIVER={{{DB_DRIVER}}};", f"SERVER={DB_SERVER};", f"DATABASE={database};"]
    if USE_WINDOWS_AUTH:
        parts.append("Trusted_Connection=yes;")
    else:
        parts.append(f"UID={DB_USERNAME};PWD={DB_PASSWORD};")
    if TRUST_SERVER_CERTIFICATE:
        parts.append("TrustServerCertificate=yes;")
    return "".join(parts)


def load_and_verify_artifact():
    artifact_path = SEED_DIR / "provisioning.v1.json"
    checksum_path = SEED_DIR / "provisioning.v1.json.sha256"
    manifest_path = SEED_DIR / "provisioning.v1.manifest.json"

    if not artifact_path.exists():
        print(f"ERROR: {artifact_path} not found. This offline release is incomplete "
              f"-- provisioning cannot proceed without it. Installation aborted.")
        sys.exit(1)
    if not checksum_path.exists() or not manifest_path.exists():
        print("ERROR: provisioning manifest/checksum files missing alongside "
              "provisioning.v1.json. Installation aborted.")
        sys.exit(1)

    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    expected_line = checksum_path.read_text(encoding="utf-8").strip()
    expected = expected_line.split()[0]
    if actual != expected:
        print(f"ERROR: checksum mismatch for provisioning.v1.json.\n"
              f"  expected: {expected}\n  actual:   {actual}\n"
              f"The release bundle may be corrupted or tampered with. Installation aborted.")
        sys.exit(1)

    manifest = json.load(open(manifest_path, encoding="utf-8"))
    artifact = json.load(open(artifact_path, encoding="utf-8"))
    print(f"Loaded and verified provisioning.v1.json (checksum OK, schema_version={manifest['schema_version']})")
    return artifact, manifest, actual


class DriftError(Exception):
    pass


def provision_org_units(cur, org_units, dry_run):
    print(f"\n--- AdminsrationUnit ({len(org_units)} units) ---")
    inserted = skipped = 0
    for u in org_units:
        cur.execute("SELECT Name, ParentID, Type FROM dbo.AdminsrationUnit WHERE UniqueID = ?", u["source_id"])
        row = cur.fetchone()
        if row is None:
            if not dry_run:
                cur.execute(
                    "INSERT INTO dbo.AdminsrationUnit (UniqueID, Name, ParentID, Frozen, Type) VALUES (?, ?, ?, 0, ?)",
                    u["source_id"], u["name"], u["parent_source_id"], u["type"],
                )
            inserted += 1
        else:
            existing = {"Name": row.Name, "ParentID": row.ParentID, "Type": row.Type}
            incoming = {"Name": u["name"], "ParentID": u["parent_source_id"], "Type": u["type"]}
            if existing != incoming:
                raise DriftError(
                    f"AdminsrationUnit {u['source_id']} drift: existing={existing} incoming={incoming}"
                )
            skipped += 1
    print(f"  inserted={inserted} skipped(identical)={skipped}")


def provision_users(cur, users, dry_run):
    print(f"\n--- APP_Users / APP_UserSourceIDMap ({len(users)} users) ---")
    inserted = reconciled = skipped = 0
    local_user_id_by_source = {}

    for u in users:
        src_id = u["source_user_id"]
        cur.execute(
            "SELECT LocalUserID FROM dbo.APP_UserSourceIDMap WHERE SourceSystem = ? AND SourceUserID = ?",
            SOURCE_SYSTEM, src_id,
        )
        row = cur.fetchone()

        if row is None:
            # Not yet mapped -- check for a username collision against an
            # unmapped account before inserting (a real conflict, not a merge).
            cur.execute("SELECT UserID FROM dbo.APP_Users WHERE Username = ?", u["username"])
            existing_user = cur.fetchone()
            if existing_user is not None:
                raise DriftError(
                    f"Username collision: {u['username']!r} already exists as UserID="
                    f"{existing_user.UserID} but is not mapped to source_user_id={src_id}. "
                    f"Two different source accounts cannot resolve to one local username."
                )
            if dry_run:
                print(f"  [dry-run] would insert new user {u['username']!r}")
                continue
            cur.execute(
                "INSERT INTO dbo.APP_Users (Username, PasswordHash, IsActive, DisplayName, DepartmentDisplayName) "
                "OUTPUT INSERTED.UserID VALUES (?, ?, ?, ?, ?)",
                u["username"], u["password_hash"], u["active"], u.get("display_name"), None,
            )
            local_user_id = cur.fetchone().UserID
            cur.execute(
                "INSERT INTO dbo.APP_UserSourceIDMap (SourceSystem, SourceUserID, LocalUserID) VALUES (?, ?, ?)",
                SOURCE_SYSTEM, src_id, local_user_id,
            )
            local_user_id_by_source[src_id] = local_user_id
            inserted += 1
        else:
            local_user_id = row.LocalUserID
            local_user_id_by_source[src_id] = local_user_id
            cur.execute(
                "SELECT Username, IsActive, DisplayName FROM dbo.APP_Users WHERE UserID = ?", local_user_id
            )
            existing = cur.fetchone()
            if existing is None:
                raise DriftError(
                    f"APP_UserSourceIDMap references LocalUserID={local_user_id} for source_user_id={src_id}, "
                    f"but no such APP_Users row exists. Data integrity problem -- not auto-repairing."
                )
            # PasswordHash is never touched on rerun. Non-security fields reconcile.
            needs_update = (bool(existing.IsActive) != u["active"]) or (existing.DisplayName != u.get("display_name"))
            if needs_update and not dry_run:
                cur.execute(
                    "UPDATE dbo.APP_Users SET IsActive = ?, DisplayName = ? WHERE UserID = ?",
                    u["active"], u.get("display_name"), local_user_id,
                )
            if needs_update:
                reconciled += 1
            else:
                skipped += 1

    print(f"  inserted={inserted} reconciled={reconciled} skipped(identical)={skipped}")
    return local_user_id_by_source


def provision_scopes(cur, users, local_user_id_by_source, dry_run):
    print(f"\n--- APP_UserRoleScope ---")
    role_id_by_code = {}
    cur.execute("SELECT RoleID, RoleCode FROM dbo.APP_Roles")
    for row in cur.fetchall():
        role_id_by_code[row.RoleCode] = row.RoleID

    inserted = skipped = no_role = 0
    for u in users:
        if not u.get("role"):
            no_role += 1
            continue
        local_user_id = local_user_id_by_source[u["source_user_id"]]
        role_id = role_id_by_code[u["role"]]
        org_unit_id = u.get("org_unit_source_id")
        org_unit_type = u.get("org_unit_type")

        cur.execute("SELECT RoleID, OrgUnitID, OrgUnitType FROM dbo.APP_UserRoleScope WHERE UserID = ?", local_user_id)
        row = cur.fetchone()
        if row is None:
            if not dry_run:
                cur.execute(
                    "INSERT INTO dbo.APP_UserRoleScope (UserID, RoleID, OrgUnitID, OrgUnitType) VALUES (?, ?, ?, ?)",
                    local_user_id, role_id, org_unit_id, org_unit_type,
                )
            inserted += 1
        else:
            existing = {"RoleID": row.RoleID, "OrgUnitID": row.OrgUnitID, "OrgUnitType": row.OrgUnitType}
            incoming = {"RoleID": role_id, "OrgUnitID": org_unit_id, "OrgUnitType": org_unit_type}
            if existing != incoming:
                raise DriftError(
                    f"APP_UserRoleScope for UserID={local_user_id} drift: existing={existing} incoming={incoming}"
                )
            skipped += 1

    print(f"  inserted={inserted} skipped(identical)={skipped} no_role_assigned={no_role}")


def provision_custom_views(cur, custom_views, local_user_id_by_source, dry_run):
    print(f"\n--- APP_CUSTOM_VIEWS / APP_CustomViewSourceIDMap ({len(custom_views)} views) ---")
    inserted = skipped = 0

    for v in custom_views:
        src_id = v["source_view_id"]
        cur.execute(
            "SELECT LocalViewID FROM dbo.APP_CustomViewSourceIDMap WHERE SourceSystem = ? AND SourceViewID = ?",
            SOURCE_SYSTEM, src_id,
        )
        row = cur.fetchone()

        show_cols = sorted(v["show_flags"].keys())
        src_created_by = v.get("source_created_by_user_id")
        # CreatedByUserID is unenforced audit metadata (matches the existing
        # APP_OrgUnitPolicy precedent) -- resolve to a local user if that
        # source user was migrated, otherwise NULL rather than a wrong guess.
        local_created_by = local_user_id_by_source.get(src_created_by) if src_created_by is not None else None

        if row is None:
            if dry_run:
                print(f"  [dry-run] would insert new custom view {v['view_name']!r}")
                continue
            col_list = ", ".join(["ViewName"] + show_cols + ["CreatedAt", "CreatedByUserID", "IsActive"])
            placeholders = ", ".join(["?"] * (4 + len(show_cols)))
            # APP_CUSTOM_VIEWS.CreatedAt is legacy `datetime` (not datetime2) --
            # passing the source's raw ISO string (with a "T" separator and
            # 6-digit microseconds) as a bound parameter fails SQL Server's
            # stricter string->datetime conversion; a real Python datetime
            # object converts cleanly instead.
            created_at_value = datetime.fromisoformat(v["created_at"]) if v.get("created_at") else None
            values = [v["view_name"]] + [v["show_flags"][c] for c in show_cols] + [created_at_value, local_created_by, v["is_active"]]
            cur.execute(
                f"INSERT INTO dbo.APP_CUSTOM_VIEWS ({col_list}) OUTPUT INSERTED.ViewID VALUES ({placeholders})",
                values,
            )
            local_view_id = cur.fetchone().ViewID
            cur.execute(
                "INSERT INTO dbo.APP_CustomViewSourceIDMap (SourceSystem, SourceViewID, LocalViewID) VALUES (?, ?, ?)",
                SOURCE_SYSTEM, src_id, local_view_id,
            )
            inserted += 1
        else:
            local_view_id = row.LocalViewID
            select_cols = ", ".join(["ViewName"] + show_cols + ["IsActive"])
            cur.execute(f"SELECT {select_cols} FROM dbo.APP_CUSTOM_VIEWS WHERE ViewID = ?", local_view_id)
            existing_row = cur.fetchone()
            if existing_row is None:
                raise DriftError(
                    f"APP_CustomViewSourceIDMap references LocalViewID={local_view_id} for source_view_id={src_id}, "
                    f"but no such APP_CUSTOM_VIEWS row exists. Data integrity problem -- not auto-repairing."
                )
            existing = dict(zip(["ViewName"] + show_cols + ["IsActive"], existing_row))
            existing_norm = {k: (bool(val) if isinstance(val, int) else val) for k, val in existing.items()}
            incoming = {"ViewName": v["view_name"], **{c: v["show_flags"][c] for c in show_cols}, "IsActive": v["is_active"]}
            if existing_norm != incoming:
                raise DriftError(
                    f"APP_CUSTOM_VIEWS ViewID={local_view_id} (source_view_id={src_id}) drift: "
                    f"existing={existing_norm} incoming={incoming}"
                )
            skipped += 1

    print(f"  inserted={inserted} skipped(identical)={skipped}")


ML_TRAINING_NON_EMBEDDING_COLUMNS = [
    "LegacySource", "LegacySourceTable", "LegacySourceRowID",
    "PossibleIncidentRequestCaseID", "LinkConfidence",
    "ComplaintText", "ImmediateActionText", "TakenActionText",
    "FeedbackTypeID", "DomainID", "CategoryID", "SubCategoryID",
    "ClassificationID", "SeverityLevelID", "StageID", "HarmLevelID",
    "ImprovementOpportunityTypeID",
    "MigrationBatchID", "PreservationNotes",
]
ML_TRAINING_EMBEDDING_COLUMNS = [
    "EmbeddingText1", "EmbeddingText2", "EmbeddingText3",
    "EmbeddingText123", "EmbeddingText23",
    "SentenceEmbedding1", "SentenceEmbedding2", "SentenceEmbedding3",
    "SentenceEmbedding4", "SentenceEmbedding5", "SentenceEmbedding6",
]


def load_and_verify_ml_training_artifact():
    """
    Optional artifact -- unlike provisioning.v1.json, a fresh install is
    fully usable without this. Training and the ML dashboards simply start
    with zero historical seed data and grow from real operational incidents
    only (the embedding worker already does this automatically -- see
    ML_ARCHITECTURE_DECISION_RECORD.md). Missing file -> skip gracefully,
    not an install failure. Present-but-corrupt -> abort, same as the
    mandatory artifact, since a tampered/truncated file is worse than none.
    """
    artifact_path = SEED_DIR / "ml_training_data.v1.json"
    checksum_path = SEED_DIR / "ml_training_data.v1.json.sha256"

    if not artifact_path.exists():
        print("\nNOTE: ml_training_data.v1.json not present -- skipping ML historical "
              "seed provisioning. 'Train All Models' and the training dashboards are "
              "fully functional without it; they will simply start empty and grow as "
              "real incidents are processed (see extract_ml_training_data.py to "
              "produce this artifact from an engineering database that already has "
              "ml.HistoricalTrainingExample populated).")
        return None

    if not checksum_path.exists():
        print("ERROR: ml_training_data.v1.json present but its checksum file is "
              "missing. Installation aborted rather than trusting an unverifiable artifact.")
        sys.exit(1)

    actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0]
    if actual != expected:
        print(f"ERROR: checksum mismatch for ml_training_data.v1.json.\n"
              f"  expected: {expected}\n  actual:   {actual}\n"
              f"The release bundle may be corrupted or tampered with. Installation aborted.")
        sys.exit(1)

    artifact = json.load(open(artifact_path, encoding="utf-8"))
    records = artifact["historical_training_examples"]
    print(f"Loaded and verified ml_training_data.v1.json (checksum OK, "
          f"{len(records)} historical training examples)")
    return records


def provision_ml_training_data(cur, records, dry_run):
    print(f"\n--- ml.HistoricalTrainingExample ({len(records)} historical examples) ---")
    inserted = skipped = 0

    col_list = ML_TRAINING_NON_EMBEDDING_COLUMNS + ML_TRAINING_EMBEDDING_COLUMNS
    placeholders = ", ".join(["?"] * len(col_list))
    insert_sql = f"INSERT INTO ml.HistoricalTrainingExample ({', '.join(col_list)}) VALUES ({placeholders})"

    for r in records:
        cur.execute(
            "SELECT 1 FROM ml.HistoricalTrainingExample WHERE LegacySourceTable = ? AND LegacySourceRowID = ?",
            r["LegacySourceTable"], r["LegacySourceRowID"],
        )
        if cur.fetchone() is not None:
            skipped += 1
            continue

        if dry_run:
            inserted += 1
            continue

        values = [r.get(col) for col in ML_TRAINING_NON_EMBEDDING_COLUMNS]
        for col in ML_TRAINING_EMBEDDING_COLUMNS:
            b64 = r.get(col)
            values.append(base64.b64decode(b64) if b64 else None)

        cur.execute(insert_sql, values)
        inserted += 1

    print(f"  inserted={inserted} skipped(already present)={skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Report what would change without writing anything.")
    args = parser.parse_args()

    artifact, manifest, checksum = load_and_verify_artifact()
    ml_training_records = load_and_verify_ml_training_artifact()

    print(f"Target: {DB_SERVER} / {DB_DATABASE}")
    if args.dry_run:
        print("*** DRY RUN -- no changes will be committed ***")

    conn = pyodbc.connect(_conn_string(DB_DATABASE), timeout=30, autocommit=False)
    cur = conn.cursor()

    try:
        provision_org_units(cur, artifact["org_units"], args.dry_run)
        local_user_id_by_source = provision_users(cur, artifact["users"], args.dry_run)
        if not args.dry_run:
            provision_scopes(cur, artifact["users"], local_user_id_by_source, args.dry_run)
            provision_custom_views(cur, artifact.get("custom_views", []), local_user_id_by_source, args.dry_run)
            if ml_training_records is not None:
                provision_ml_training_data(cur, ml_training_records, args.dry_run)
        else:
            # Still validate scope logic in dry-run, using a placeholder map
            # (no real LocalUserIDs exist yet for brand-new users in dry-run).
            placeholder_map = {u["source_user_id"]: local_user_id_by_source.get(u["source_user_id"], -1)
                                for u in artifact["users"]}
            print("\n--- APP_UserRoleScope --- (skipped in dry-run: depends on IDs not yet assigned)")
            provision_custom_views(cur, artifact.get("custom_views", []), placeholder_map, args.dry_run)
            if ml_training_records is not None:
                provision_ml_training_data(cur, ml_training_records, args.dry_run)

        if not args.dry_run:
            migration_name = f"provisioning_{SOURCE_SYSTEM}"
            cur.execute("SELECT 1 FROM dbo.SchemaMigrationHistory WHERE MigrationName = ?", migration_name)
            if cur.fetchone() is None:
                cur.execute(
                    "INSERT INTO dbo.SchemaMigrationHistory (MigrationName, Checksum, AppliedAt, AppliedBy, ApplicationVersion, Success) "
                    "VALUES (?, ?, SYSUTCDATETIME(), SUSER_SNAME(), ?, 1)",
                    migration_name, checksum, "1.0.0",
                )
            conn.commit()
            print("\n=== Provisioning committed successfully ===")
        else:
            conn.rollback()
            print("\n=== Dry run complete -- rolled back, nothing committed ===")

    except DriftError as e:
        conn.rollback()
        print(f"\n=== DRIFT DETECTED -- provisioning FAILED, rolled back entirely ===\n{e}")
        sys.exit(1)
    except Exception as e:
        conn.rollback()
        print(f"\n=== ERROR -- provisioning FAILED, rolled back entirely ===\n{e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
