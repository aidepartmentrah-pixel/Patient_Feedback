"""
ML Architecture Consolidation — Stage 8: Historical Migration

Classifies every row in the legacy SQLite ML store (patient_feedback_encoded,
patient_feedback_encoded_Old) by match-confidence against the live
dbo.APP_IncidentCase table, then migrates them into the new SQL Server `ml`
schema:

  - Every non-excluded row (any tier) -> one ml.HistoricalTrainingExample row,
    all 11 legacy embedding columns copied as raw bytes exactly as authored,
    tagged with a single MigrationBatchID for this run.
  - Exact/High tier rows ALSO get an upsert into ml.CaseTrainingRecord
    (text + labels only, never legacy embedding bytes) plus a
    'MigrationBackfill' ml.EmbeddingProcessingJob so the existing Stage 6
    worker recomputes embeddings fresh through the live, version-tracked
    model path — but ONLY if that case doesn't already have a
    ml.CaseTrainingRecord row (a live row is always more current than a
    legacy SQLite guess and must never be overwritten by it).

Rows that are obviously synthetic test/dev fixtures (see EXCLUSION_MARKERS)
are excluded from migration entirely and reported separately — they are not
"historical human-entered data" and would pollute the training corpus.

Defaults to --dry-run: prints the exclusion list and tier-count summary,
writes nothing. Pass --apply to actually write. Safe to re-run --apply: a
per-row pre-check plus the UQ_ml_HistoricalTrainingExample_LegacySource
unique index (see phase_ml_s8_historical_migration_schema.sql) make this
idempotent — nothing is inserted twice.

Run from the backend/ directory:
    python -m scripts.ml_stage8_historical_migration              # dry run
    python -m scripts.ml_stage8_historical_migration --apply       # writes

Never writes to or deletes the source SQLite file — read-only against it.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from core.database import get_connection
from api.db_layer import ml_case_training_db, ml_embedding_job_db

SQLITE_PATH = os.path.join(_REPO_ROOT, "models_directory", "patient_feedback_ml.db")
ARCHIVE_ROOT = r"C:\SQLBackup"

SOURCE_TABLES = ["patient_feedback_encoded", "patient_feedback_encoded_Old"]

SQLITE_COLUMNS = [
    "id", "feedback_received_date", "feedback_type", "domain", "category", "sub_category",
    "classification_en", "complaint_text", "immediate_action", "taken_action",
    "severity_level", "stage", "harm_level", "improvement_opportunity_type",
    "embedding_text1", "embedding_text2", "embedding_text3", "embedding_text123", "embedding_text23",
    "sentence_1_embedding", "sentence_2_embedding", "sentence_3_embedding",
    "sentence_4_embedding", "sentence_5_embedding", "sentence_6_embedding",
]

# Evidence-based, not a broad "contains TEST" catch-all — these are the exact
# markers this codebase's own smoke tests and ad hoc dev testing use today
# (confirmed by direct inspection during Stage 8 planning). A row this filter
# misses is low-risk anyway: with no matching live case, it will correctly
# fall to Unmatched rather than corrupting an Exact/High link.
EXCLUSION_MARKERS = ["smoke test", "test session", "safe to delete"]
EXCLUSION_PREFIXES = ["test complaint text"]

POSSIBLE_THRESHOLD = 0.85
CONFLICT_MARGIN = 0.05


def is_test_artifact(complaint_text):
    if not complaint_text:
        return False
    t = complaint_text.strip().lower()
    if any(t.startswith(p) for p in EXCLUSION_PREFIXES):
        return True
    return any(marker in t for marker in EXCLUSION_MARKERS)


def normalize_text(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_sqlite_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def dates_match(sqlite_date_str, case_date):
    d1 = parse_sqlite_date(sqlite_date_str)
    if d1 is None or case_date is None:
        return False
    return d1 == case_date


def load_candidates(cursor):
    cursor.execute(
        """
        SELECT IncidentRequestCaseID, ComplaintText, FeedbackRecievedDate,
               DomainID, CategoryID, SubCategoryID, ClassificationID, SeverityID, StageID, HarmLevelID
        FROM dbo.APP_IncidentCase
        """
    )
    candidates = []
    for row in cursor.fetchall():
        candidates.append({
            "case_id": row.IncidentRequestCaseID,
            "norm_text": normalize_text(row.ComplaintText),
            "feedback_date": row.FeedbackRecievedDate,
            "labels": {
                "DomainID": row.DomainID,
                "CategoryID": row.CategoryID,
                "SubCategoryID": row.SubCategoryID,
                "ClassificationID": row.ClassificationID,
                "SeverityLevelID": row.SeverityID,
                "StageID": row.StageID,
                "HarmLevelID": row.HarmLevelID,
            },
        })
    return candidates


def classify_row(norm_text, feedback_date_str, candidates):
    """Returns (tier, matched_case_id_or_None, ratio, label_mismatches)."""
    if not norm_text:
        return "Unmatched", None, 0.0, []

    exact_candidates = [c for c in candidates if c["norm_text"] == norm_text]
    if len(exact_candidates) == 1:
        candidate = exact_candidates[0]
        tier = "Exact" if dates_match(feedback_date_str, candidate["feedback_date"]) else "High"
        return tier, candidate["case_id"], 1.0, candidate
    if len(exact_candidates) > 1:
        return "Conflict", None, 1.0, None

    scored = sorted(
        ((SequenceMatcher(None, norm_text, c["norm_text"]).ratio(), c) for c in candidates if c["norm_text"]),
        key=lambda x: -x[0],
    )
    if not scored:
        return "Unmatched", None, 0.0, None

    best_ratio, best = scored[0]
    second_ratio = scored[1][0] if len(scored) > 1 else 0.0

    if best_ratio >= POSSIBLE_THRESHOLD and (best_ratio - second_ratio) >= CONFLICT_MARGIN:
        return "Possible", best["case_id"], best_ratio, best
    if best_ratio >= POSSIBLE_THRESHOLD:
        return "Conflict", None, best_ratio, None
    return "Unmatched", None, best_ratio, None


def label_mismatches(sqlite_row, candidate):
    if candidate is None:
        return []
    mapping = {
        "domain": "DomainID", "category": "CategoryID", "sub_category": "SubCategoryID",
        "classification_en": "ClassificationID", "severity_level": "SeverityLevelID",
        "stage": "StageID", "harm_level": "HarmLevelID",
    }
    mismatches = []
    for sqlite_col, target_col in mapping.items():
        sv, cv = sqlite_row.get(sqlite_col), candidate["labels"].get(target_col)
        if sv is not None and cv is not None and int(sv) != int(cv):
            mismatches.append(f"{target_col}: legacy={sv} vs case={cv}")
    return mismatches


def fetch_sqlite_rows(table):
    """
    `id` is a plain nullable INTEGER column in these tables, not an actual
    primary key — patient_feedback_encoded_Old has 7 rows where it's
    genuinely NULL. sqlite's own `rowid` is always present and unique per
    row, but raw rowid values collide with real `id` values used by OTHER
    rows in the same table (confirmed during Stage 8 verification), so a
    row with NULL `id` gets `_row_identity` = -rowid instead: real ids are
    always positive, so the negative space can never collide.
    """
    conn = sqlite3.connect(f"file:{SQLITE_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT rowid, {','.join(SQLITE_COLUMNS)} FROM {table}")
    rows = []
    for row in cur.fetchall():
        d = dict(row)
        d["_row_identity"] = d["id"] if d["id"] is not None else -d["rowid"]
        rows.append(d)
    conn.close()
    return rows


EMBEDDING_COLUMN_MAP = {
    "embedding_text1": "EmbeddingText1", "embedding_text2": "EmbeddingText2",
    "embedding_text3": "EmbeddingText3", "embedding_text123": "EmbeddingText123",
    "embedding_text23": "EmbeddingText23",
    "sentence_1_embedding": "SentenceEmbedding1", "sentence_2_embedding": "SentenceEmbedding2",
    "sentence_3_embedding": "SentenceEmbedding3", "sentence_4_embedding": "SentenceEmbedding4",
    "sentence_5_embedding": "SentenceEmbedding5", "sentence_6_embedding": "SentenceEmbedding6",
}

LABEL_COLUMN_MAP = {
    "feedback_type": "FeedbackTypeID", "domain": "DomainID", "category": "CategoryID",
    "sub_category": "SubCategoryID", "classification_en": "ClassificationID",
    "severity_level": "SeverityLevelID", "stage": "StageID", "harm_level": "HarmLevelID",
    "improvement_opportunity_type": "ImprovementOpportunityTypeID",
}


def already_migrated(cursor, source_table, row_id):
    cursor.execute(
        "SELECT 1 FROM ml.HistoricalTrainingExample WHERE LegacySourceTable = ? AND LegacySourceRowID = ?",
        (source_table, row_id),
    )
    return cursor.fetchone() is not None


def insert_historical_example(cursor, source_table, row, tier, ratio, candidate, mismatches, batch_id):
    notes = f"Stage 8 migration. Tier={tier}, ratio={ratio:.3f}."
    if candidate:
        notes += f" Matched case {candidate['case_id']}."
    if mismatches:
        notes += " Label mismatches: " + "; ".join(mismatches)
    if row["id"] is None:
        notes += f" (legacy id was NULL; using synthetic identity {row['_row_identity']} = -rowid)"

    columns = ["LegacySource", "LegacySourceTable", "LegacySourceRowID",
               "PossibleIncidentRequestCaseID", "LinkConfidence",
               "ComplaintText", "ImmediateActionText", "TakenActionText",
               "MigrationBatchID", "PreservationNotes"]
    values = ["SQLite:patient_feedback_ml.db", source_table, row["_row_identity"],
              candidate["case_id"] if candidate else None, tier,
              row.get("complaint_text"), row.get("immediate_action"), row.get("taken_action"),
              batch_id, notes]

    for sqlite_col, target_col in LABEL_COLUMN_MAP.items():
        columns.append(target_col)
        values.append(row.get(sqlite_col))

    for sqlite_col, target_col in EMBEDDING_COLUMN_MAP.items():
        columns.append(target_col)
        values.append(row.get(sqlite_col))

    placeholders = ",".join(["?"] * len(values))
    cursor.execute(
        f"INSERT INTO ml.HistoricalTrainingExample ({','.join(columns)}) VALUES ({placeholders})",
        values,
    )


def migrate_case_training_record(cursor, case_id, row):
    if ml_case_training_db.get_case_training_record(cursor, case_id) is not None:
        return False  # live row already exists — never overwrite with legacy data

    fields = {
        "ComplaintText": row.get("complaint_text"),
        "ImmediateActionText": row.get("immediate_action"),
        "TakenActionText": row.get("taken_action"),
    }
    for sqlite_col, target_col in LABEL_COLUMN_MAP.items():
        if row.get(sqlite_col) is not None:
            fields[target_col] = row.get(sqlite_col)

    ml_case_training_db.upsert_case_training_record(cursor, case_id, fields)
    ml_embedding_job_db.insert_embedding_job(cursor, case_id, "MigrationBackfill")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually write; default is dry-run.")
    args = parser.parse_args()

    print("=" * 70)
    print("ML ARCHITECTURE CONSOLIDATION — STAGE 8: HISTORICAL MIGRATION")
    print(f"Mode: {'APPLY (writing)' if args.apply else 'DRY RUN (no writes)'}")
    print("=" * 70)

    conn = get_connection()
    cursor = conn.cursor()

    print("\n[1] Fresh inventory")
    cursor.execute("SELECT COUNT(*) FROM dbo.APP_IncidentCase")
    print(f"    dbo.APP_IncidentCase: {cursor.fetchone()[0]} rows")
    candidates = load_candidates(cursor)
    print(f"    Loaded {len(candidates)} candidate cases for matching")

    batch_id = f"ML_S8_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    print(f"    MigrationBatchID for this run: {batch_id}")

    tier_counts = {"Exact": 0, "High": 0, "Possible": 0, "Conflict": 0, "Unmatched": 0}
    excluded_rows = []
    already_migrated_count = 0
    case_training_written = 0
    case_training_skipped_existing = 0
    per_table_counts = {}

    for source_table in SOURCE_TABLES:
        rows = fetch_sqlite_rows(source_table)
        print(f"\n[2] {source_table}: {len(rows)} rows")
        table_tier_counts = {"Exact": 0, "High": 0, "Possible": 0, "Conflict": 0, "Unmatched": 0}
        table_excluded = 0

        for row in rows:
            if is_test_artifact(row.get("complaint_text")):
                excluded_rows.append((source_table, row["id"], (row.get("complaint_text") or "")[:60]))
                table_excluded += 1
                continue

            if args.apply and already_migrated(cursor, source_table, row["_row_identity"]):
                already_migrated_count += 1
                continue

            norm_text = normalize_text(row.get("complaint_text"))
            tier, matched_case_id, ratio, candidate = classify_row(
                norm_text, row.get("feedback_received_date"), candidates
            )
            mismatches = label_mismatches(row, candidate) if candidate else []

            tier_counts[tier] += 1
            table_tier_counts[tier] += 1

            if args.apply:
                insert_historical_example(cursor, source_table, row, tier, ratio, candidate, mismatches, batch_id)
                if tier in ("Exact", "High") and matched_case_id is not None:
                    if migrate_case_training_record(cursor, matched_case_id, row):
                        case_training_written += 1
                    else:
                        case_training_skipped_existing += 1

        per_table_counts[source_table] = {"tiers": table_tier_counts, "excluded": table_excluded}
        if args.apply:
            conn.commit()

    print("\n[3] Excluded test/dev artifacts (not migrated):")
    for source_table, row_id, text_preview in excluded_rows:
        print(f"    {source_table} id={row_id}: {text_preview!r}")
    print(f"    Total excluded: {len(excluded_rows)}")

    print("\n[4] Match-confidence tier summary:")
    for tier, count in tier_counts.items():
        print(f"    {tier}: {count}")
    if args.apply:
        print(f"    Already migrated (skipped, idempotent re-run): {already_migrated_count}")
        print(f"    ml.CaseTrainingRecord rows written (Exact/High, no live row existed): {case_training_written}")
        print(f"    ml.CaseTrainingRecord skipped (live row already existed): {case_training_skipped_existing}")

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "mode": "apply" if args.apply else "dry_run",
        "migration_batch_id": batch_id,
        "candidate_case_count": len(candidates),
        "tier_counts": tier_counts,
        "per_table_counts": per_table_counts,
        "excluded_count": len(excluded_rows),
        "excluded_rows": [{"source_table": t, "row_id": r, "text_preview": p} for t, r, p in excluded_rows],
        "already_migrated_count": already_migrated_count,
        "case_training_written": case_training_written,
        "case_training_skipped_existing": case_training_skipped_existing,
    }
    os.makedirs(ARCHIVE_ROOT, exist_ok=True)
    report_suffix = "apply" if args.apply else "dryrun"
    report_path = os.path.join(
        ARCHIVE_ROOT, f"ml_stage8_migration_report_{report_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport written to: {report_path}")

    cursor.close()
    conn.close()

    if not args.apply:
        print("\nDry run complete — nothing was written. Review the exclusion list and tier")
        print("counts above, then re-run with --apply once satisfied.")
    print("=" * 70)


if __name__ == "__main__":
    main()
