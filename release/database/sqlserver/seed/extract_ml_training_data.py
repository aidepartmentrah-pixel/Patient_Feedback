"""
ML historical training-data seed -- extracts ml.HistoricalTrainingExample from
the current SQL Server database into a standalone, gitignored, checksummed
provisioning artifact (ml_training_data.v1.json), so a fresh offline install
starts with the same historical training corpus this engineering database
already has, instead of an empty ml schema.

Why this exists: ml.HistoricalTrainingExample was originally populated by
running scripts/ml_stage8_historical_migration.py against the retired legacy
SQLite store (models_directory/patient_feedback_ml.db, officially retired
2026-07-20 -- see ML_ARCHITECTURE_DECISION_RECORD.md section 13). That
migration is a one-time, source-machine-specific operation (it needs the
retired SQLite file, which must never travel again). This script instead
captures its OUTPUT -- the resulting SQL rows -- so the offline installer
never needs the SQLite file at all; it just needs this JSON artifact.

Same governance as provisioning.v1.json: contains real patient complaint
text, so it is gitignored, never committed, shipped only as a physically
transferred file alongside the release bundle, checksummed so tampering /
corruption in transit is detectable.

Precomputed embedding columns (varbinary) are base64-encoded for JSON
transport and decoded back to raw bytes at provisioning time.

Usage:
    python database/sqlserver/seed/extract_ml_training_data.py
"""
import base64
import hashlib
import json
import sys
from datetime import datetime, timezone
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

EMBEDDING_COLUMNS = [
    "EmbeddingText1", "EmbeddingText2", "EmbeddingText3",
    "EmbeddingText123", "EmbeddingText23",
    "SentenceEmbedding1", "SentenceEmbedding2", "SentenceEmbedding3",
    "SentenceEmbedding4", "SentenceEmbedding5", "SentenceEmbedding6",
]

NON_EMBEDDING_COLUMNS = [
    "LegacySource", "LegacySourceTable", "LegacySourceRowID",
    "PossibleIncidentRequestCaseID", "LinkConfidence",
    "ComplaintText", "ImmediateActionText", "TakenActionText",
    "FeedbackTypeID", "DomainID", "CategoryID", "SubCategoryID",
    "ClassificationID", "SeverityLevelID", "StageID", "HarmLevelID",
    "ImprovementOpportunityTypeID",
    "MigrationBatchID", "PreservationNotes",
]

ALL_COLUMNS = NON_EMBEDDING_COLUMNS + EMBEDDING_COLUMNS


def _conn_string(database: str) -> str:
    parts = [f"DRIVER={{{DB_DRIVER}}};", f"SERVER={DB_SERVER};", f"DATABASE={database};"]
    if USE_WINDOWS_AUTH:
        parts.append("Trusted_Connection=yes;")
    else:
        parts.append(f"UID={DB_USERNAME};PWD={DB_PASSWORD};")
    if TRUST_SERVER_CERTIFICATE:
        parts.append("TrustServerCertificate=yes;")
    return "".join(parts)


def main():
    print("=== Extracting ml.HistoricalTrainingExample seed data ===\n")
    print(f"Source: {DB_SERVER} / {DB_DATABASE}")

    conn = pyodbc.connect(_conn_string(DB_DATABASE), timeout=30)
    cur = conn.cursor()

    col_list = ", ".join(ALL_COLUMNS)
    cur.execute(f"SELECT {col_list} FROM ml.HistoricalTrainingExample ORDER BY HistoricalTrainingExampleID")
    rows = cur.fetchall()
    conn.close()

    print(f"Fetched {len(rows)} rows")

    records = []
    embedding_present_counts = {c: 0 for c in EMBEDDING_COLUMNS}
    for row in rows:
        rec = {}
        for col in NON_EMBEDDING_COLUMNS:
            val = getattr(row, col)
            if isinstance(val, datetime):
                val = val.isoformat()
            rec[col] = val
        for col in EMBEDDING_COLUMNS:
            val = getattr(row, col)
            if val is not None:
                rec[col] = base64.b64encode(bytes(val)).decode("ascii")
                embedding_present_counts[col] += 1
            else:
                rec[col] = None
        records.append(rec)

    OUT_DIR = SEED_DIR
    final_path = OUT_DIR / "ml_training_data.v1.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"historical_training_examples": records}, f, ensure_ascii=False, indent=2)
    tmp_path.replace(final_path)
    print(f"\nWrote {final_path}")

    # ---- Validate: reload and check counts match ----
    reloaded = json.load(open(final_path, encoding="utf-8"))
    assert len(reloaded["historical_training_examples"]) == len(records), "round-trip count mismatch"
    print(f"Validation: round-trip record count matches ({len(records)}).")

    # ---- External manifest + checksum ----
    file_bytes = final_path.read_bytes()
    checksum = hashlib.sha256(file_bytes).hexdigest()

    link_confidence_counts = {}
    for r in records:
        lc = r.get("LinkConfidence") or "(none)"
        link_confidence_counts[lc] = link_confidence_counts.get(lc, 0) + 1

    manifest = {
        "schema_version": 1,
        "source": "ml.HistoricalTrainingExample (local SQL Server, produced by "
                   "scripts/ml_stage8_historical_migration.py against the retired "
                   "legacy SQLite store -- see ML_ARCHITECTURE_DECISION_RECORD.md section 13)",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "checksum_sha256": checksum,
        "record_counts": {
            "total": len(records),
            "by_link_confidence": link_confidence_counts,
            "embedding_columns_non_null": embedding_present_counts,
        },
    }
    manifest_path = OUT_DIR / "ml_training_data.v1.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Wrote {manifest_path}")

    sha256_path = OUT_DIR / "ml_training_data.v1.json.sha256"
    with open(sha256_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{checksum}  ml_training_data.v1.json\n")
    print(f"Wrote {sha256_path}")

    print("\n=== Extraction report ===")
    print(json.dumps(manifest["record_counts"], indent=2, ensure_ascii=False))
    print(f"\nFile size: {len(file_bytes) / (1024*1024):.1f} MB")
    print("\nThis contains real patient complaint text -- handle with the same care as "
          "provisioning.v1.json. Gitignored, never commit, ship only as a physically "
          "transferred file alongside the release bundle.")


if __name__ == "__main__":
    main()
