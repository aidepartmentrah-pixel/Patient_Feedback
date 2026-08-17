"""
ML Import Batch / Source Record Map — DB Layer
Pure SQL operations only. No business logic.

Generalizes the duplicate-prevention pattern already proven in
dbo.APP_DataMigration_Map (Phase K legacy migration) for the bulk-import
pipeline: a durable external identifier, a mapping to the created case,
and a uniqueness constraint checked before insert.
"""

from typing import Any, Dict, List, Optional


def create_import_batch(
    cursor,
    original_file_name: Optional[str],
    file_checksum: Optional[str],
    template_version: Optional[str],
    uploaded_by_user_id: Optional[int],
) -> int:
    cursor.execute(
        """
        INSERT INTO ml.ImportBatch (OriginalFileName, FileChecksum, TemplateVersion, UploadedByUserID)
        OUTPUT INSERTED.ImportBatchID
        VALUES (?, ?, ?, ?)
        """,
        (original_file_name, file_checksum, template_version, uploaded_by_user_id),
    )
    return cursor.fetchone()[0]


def find_batch_by_checksum(cursor, file_checksum: str) -> Optional[Dict[str, Any]]:
    """Batch-level duplicate check: has this exact file already been uploaded (and completed)?"""
    cursor.execute(
        "SELECT TOP 1 * FROM ml.ImportBatch WHERE FileChecksum = ? AND Status = 'Completed' "
        "ORDER BY UploadedAt DESC",
        (file_checksum,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [c[0] for c in cursor.description]
    return dict(zip(columns, row))


def get_batch(cursor, import_batch_id: int) -> Optional[Dict[str, Any]]:
    """Fetch one batch row by id, or None if it doesn't exist."""
    cursor.execute("SELECT * FROM ml.ImportBatch WHERE ImportBatchID = ?", (import_batch_id,))
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [c[0] for c in cursor.description]
    return dict(zip(columns, row))


def delete_batch(cursor, import_batch_id: int) -> None:
    """Delete a batch row (only ever called for PendingReview/Processing
    batches -- a Completed batch's ImportSourceRecordMap rows reference
    real cases and must never be deleted this way)."""
    cursor.execute("DELETE FROM ml.ImportBatch WHERE ImportBatchID = ?", (import_batch_id,))


def list_batches(cursor, limit: int = 50) -> List[Dict[str, Any]]:
    """Most recent import batches, newest first -- backs the batch history page."""
    cursor.execute(
        "SELECT TOP (?) * FROM ml.ImportBatch ORDER BY UploadedAt DESC",
        (limit,),
    )
    rows = cursor.fetchall()
    columns = [c[0] for c in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def update_batch_summary(cursor, import_batch_id: int, **counts) -> None:
    """
    counts may include any of: total_rows, accepted_rows, rejected_rows,
    duplicate_rows, created_case_count, ml_completed_count, ml_failed_count, status
    """
    column_map = {
        "total_rows": "TotalRows", "accepted_rows": "AcceptedRows", "rejected_rows": "RejectedRows",
        "duplicate_rows": "DuplicateRows", "created_case_count": "CreatedCaseCount",
        "ml_completed_count": "MLCompletedCount", "ml_failed_count": "MLFailedCount", "status": "Status",
    }
    set_clauses = []
    values = []
    for key, value in counts.items():
        if key in column_map:
            set_clauses.append(f"{column_map[key]} = ?")
            values.append(value)

    if not set_clauses:
        return

    if counts.get("status") == "Completed":
        set_clauses.append("CompletedAt = GETDATE()")

    values.append(import_batch_id)
    cursor.execute(
        f"UPDATE ml.ImportBatch SET {','.join(set_clauses)} WHERE ImportBatchID = ?",
        values,
    )


def find_source_record_map(
    cursor, external_source_system: str, external_record_id: str
) -> Optional[int]:
    """
    Record-level duplicate check for one specific external record ID.
    Returns the existing IncidentRequestCaseID if already imported, else None.
    """
    cursor.execute(
        "SELECT IncidentRequestCaseID FROM ml.ImportSourceRecordMap "
        "WHERE ExternalSourceSystem = ? AND ExternalRecordID = ?",
        (external_source_system, external_record_id),
    )
    row = cursor.fetchone()
    return row[0] if row else None


def record_source_map(
    cursor,
    import_batch_id: Optional[int],
    external_source_system: str,
    external_record_id: str,
    incident_request_case_id: int,
) -> int:
    cursor.execute(
        """
        INSERT INTO ml.ImportSourceRecordMap
            (ImportBatchID, ExternalSourceSystem, ExternalRecordID, IncidentRequestCaseID)
        OUTPUT INSERTED.ImportSourceRecordMapID
        VALUES (?, ?, ?, ?)
        """,
        (import_batch_id, external_source_system, external_record_id, incident_request_case_id),
    )
    return cursor.fetchone()[0]
