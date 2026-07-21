"""
ML Case Training Record — DB Layer
Pure SQL operations only. No business logic.

ml.CaseTrainingRecord holds at most one current row per operational case
(UNIQUE constraint on IncidentRequestCaseID) — these functions implement
that as a real upsert, replacing the old SQLite adapter's append-only insert.
"""

from typing import Any, Dict, Optional


LABEL_COLUMNS = (
    "FeedbackTypeID", "DomainID", "CategoryID", "SubCategoryID", "ClassificationID",
    "SeverityLevelID", "StageID", "HarmLevelID", "ImprovementOpportunityTypeID",
)

TEXT_COLUMNS = ("ComplaintText", "ImmediateActionText", "TakenActionText")


def upsert_case_training_record(cursor, incident_request_case_id: int, fields: Dict[str, Any]) -> int:
    """
    Insert or update the one current ml.CaseTrainingRecord row for a case.

    fields may contain any of TEXT_COLUMNS / LABEL_COLUMNS — only columns
    actually present in `fields` are written; anything else on the row is
    left untouched. Sets ProcessingStatus='Pending' and SourceDataUpdatedAt
    on any write so the worker knows this row now needs (re)processing.

    Returns the CaseTrainingRecordID.
    """
    known_columns = TEXT_COLUMNS + LABEL_COLUMNS
    present = {k: v for k, v in fields.items() if k in known_columns}

    cursor.execute(
        "SELECT CaseTrainingRecordID FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?",
        (incident_request_case_id,),
    )
    existing = cursor.fetchone()

    if existing is None:
        columns = ["IncidentRequestCaseID"] + list(present.keys()) + ["ProcessingStatus", "SourceDataUpdatedAt"]
        placeholders = ",".join(["?"] * (len(present) + 1) + ["?", "GETDATE()"])
        values = [incident_request_case_id] + list(present.values()) + ["Pending"]

        cursor.execute(
            f"""
            INSERT INTO ml.CaseTrainingRecord ({",".join(columns)})
            OUTPUT INSERTED.CaseTrainingRecordID
            VALUES ({placeholders})
            """,
            values,
        )
        return cursor.fetchone()[0]

    record_id = existing[0]
    if present:
        set_clauses = [f"{col} = ?" for col in present.keys()]
        set_clauses += ["ProcessingStatus = ?", "SourceDataUpdatedAt = GETDATE()", "UpdatedAt = GETDATE()"]
        values = list(present.values()) + ["Pending", record_id]

        cursor.execute(
            f"""
            UPDATE ml.CaseTrainingRecord
            SET {",".join(set_clauses)}
            WHERE CaseTrainingRecordID = ?
            """,
            values,
        )
    return record_id


def get_case_training_record(cursor, incident_request_case_id: int) -> Optional[Dict[str, Any]]:
    cursor.execute(
        "SELECT * FROM ml.CaseTrainingRecord WHERE IncidentRequestCaseID = ?",
        (incident_request_case_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [c[0] for c in cursor.description]
    return dict(zip(columns, row))


def update_embeddings(
    cursor,
    incident_request_case_id: int,
    complaint_embedding: Optional[bytes],
    combined_embedding: Optional[bytes],
    embedding_model_version_id: Optional[int],
    embedding_dimension: Optional[int],
) -> None:
    """Worker calls this once embeddings for a case are computed."""
    cursor.execute(
        """
        UPDATE ml.CaseTrainingRecord
        SET ComplaintEmbedding = ?,
            CombinedTextEmbedding = ?,
            EmbeddingModelVersionID = ?,
            EmbeddingDimension = ?,
            ProcessingStatus = 'Completed',
            LastProcessedAt = GETDATE(),
            UpdatedAt = GETDATE()
        WHERE IncidentRequestCaseID = ?
        """,
        (complaint_embedding, combined_embedding, embedding_model_version_id, embedding_dimension,
         incident_request_case_id),
    )
