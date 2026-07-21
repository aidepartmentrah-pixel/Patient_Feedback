"""
ML Embedding Processing Job — DB Layer
Pure SQL operations only. No business logic.

All functions accept an open cursor (same convention as import_db.py) so a
caller can register a job in the same transaction as the operational case
insert it belongs to.
"""

from typing import Any, Dict, List, Optional


def insert_embedding_job(
    cursor,
    incident_request_case_id: int,
    job_type: str,
    import_batch_id: Optional[int] = None,
) -> int:
    """
    Register a new ML processing job. job_type must be one of:
    Create, Reprocess, TextChanged, LabelsChanged, ModelUpgrade, MigrationBackfill
    """
    cursor.execute(
        """
        INSERT INTO ml.EmbeddingProcessingJob (IncidentRequestCaseID, JobType, ImportBatchID)
        OUTPUT INSERTED.EmbeddingProcessingJobID
        VALUES (?, ?, ?)
        """,
        (incident_request_case_id, job_type, import_batch_id),
    )
    return cursor.fetchone()[0]


def claim_pending_jobs(cursor, batch_size: int) -> List[Dict[str, Any]]:
    """
    Claim up to batch_size Pending/RetryPending jobs by marking them Processing
    and returning their data. Not concurrency-hardened beyond a single worker
    (see ML_ARCHITECTURE_DECISION_RECORD.md — one in-process worker by design).
    """
    cursor.execute(
        """
        SELECT TOP (?) EmbeddingProcessingJobID, IncidentRequestCaseID, JobType, AttemptCount
        FROM ml.EmbeddingProcessingJob
        WHERE Status IN ('Pending', 'RetryPending')
          AND (NextRetryAt IS NULL OR NextRetryAt <= GETDATE())
        ORDER BY RequestedAt ASC
        """,
        (batch_size,),
    )
    rows = cursor.fetchall()
    jobs = [
        {
            "job_id": row.EmbeddingProcessingJobID,
            "incident_request_case_id": row.IncidentRequestCaseID,
            "job_type": row.JobType,
            "attempt_count": row.AttemptCount,
        }
        for row in rows
    ]

    if jobs:
        ids = [j["job_id"] for j in jobs]
        placeholders = ",".join("?" * len(ids))
        cursor.execute(
            f"UPDATE ml.EmbeddingProcessingJob SET Status = 'Processing', StartedAt = GETDATE() "
            f"WHERE EmbeddingProcessingJobID IN ({placeholders})",
            ids,
        )

    return jobs


def mark_job_completed(cursor, job_id: int, embedding_model_version_id: Optional[int]) -> None:
    cursor.execute(
        """
        UPDATE ml.EmbeddingProcessingJob
        SET Status = 'Completed', CompletedAt = GETDATE(), EmbeddingModelVersionID = ?
        WHERE EmbeddingProcessingJobID = ?
        """,
        (embedding_model_version_id, job_id),
    )


def mark_job_failed(
    cursor,
    job_id: int,
    error_code: str,
    error_message: str,
    max_attempts: int = 5,
    retry_delay_minutes: int = 5,
) -> None:
    """
    Increments AttemptCount. If under MaximumAttempts, schedules a retry;
    otherwise marks permanently Failed.
    """
    cursor.execute(
        "SELECT AttemptCount, MaximumAttempts FROM ml.EmbeddingProcessingJob WHERE EmbeddingProcessingJobID = ?",
        (job_id,),
    )
    row = cursor.fetchone()
    attempt_count = (row.AttemptCount if row else 0) + 1
    max_allowed = row.MaximumAttempts if row else max_attempts

    if attempt_count >= max_allowed:
        cursor.execute(
            """
            UPDATE ml.EmbeddingProcessingJob
            SET Status = 'Failed', AttemptCount = ?, LastErrorCode = ?, LastErrorMessage = ?
            WHERE EmbeddingProcessingJobID = ?
            """,
            (attempt_count, error_code, error_message, job_id),
        )
    else:
        cursor.execute(
            """
            UPDATE ml.EmbeddingProcessingJob
            SET Status = 'RetryPending', AttemptCount = ?, LastErrorCode = ?, LastErrorMessage = ?,
                NextRetryAt = DATEADD(MINUTE, ?, GETDATE())
            WHERE EmbeddingProcessingJobID = ?
            """,
            (attempt_count, error_code, error_message, retry_delay_minutes, job_id),
        )


def sweep_stuck_jobs(cursor, stuck_threshold_minutes: int = 15) -> int:
    """
    Recover jobs left in 'Processing' beyond the threshold (e.g. after a
    backend restart mid-batch) by moving them back to RetryPending.
    Returns the number of jobs recovered.
    """
    cursor.execute(
        """
        UPDATE ml.EmbeddingProcessingJob
        SET Status = 'RetryPending', NextRetryAt = GETDATE()
        WHERE Status = 'Processing'
          AND StartedAt < DATEADD(MINUTE, -?, GETDATE())
        """,
        (stuck_threshold_minutes,),
    )
    return cursor.rowcount
