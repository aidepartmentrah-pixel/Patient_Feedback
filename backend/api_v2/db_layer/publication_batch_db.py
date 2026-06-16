"""
Publication Batch Database Layer (API V2)
Handles SQL operations for APP_PublicationBatch / APP_PublicationBatchCase.

This is the foundation for HCAT Performance & Delay Monitoring (Stage 2),
Session 1 - Publication Batch Tracking.

NO business logic. NO authorization. ONLY SQL operations.
"""

from typing import Any, Dict, List, Optional
from core.database import get_connection


def create_publication_batch(cursor, published_by_user_id: int, cases_published_count: int) -> Dict[str, Any]:
    """
    Create a new APP_PublicationBatch row using the caller's open cursor/transaction.

    Generates the next sequential PublicationSerial as MAX(PublicationSerial) + 1
    under WITH (UPDLOCK, HOLDLOCK) so concurrent Publish All operations cannot
    receive duplicate/skipped serials. The caller owns the transaction and must
    commit (or rollback) afterwards.

    Returns:
        {"publication_batch_id": int, "publication_serial": int, "published_at": datetime}
    """
    cursor.execute(
        "SELECT ISNULL(MAX(PublicationSerial), 0) + 1 FROM dbo.APP_PublicationBatch WITH (UPDLOCK, HOLDLOCK)"
    )
    next_serial = cursor.fetchone()[0]

    cursor.execute(
        """
        INSERT INTO dbo.APP_PublicationBatch (
            PublicationSerial, PublishedAt, PublishedByUserID, CasesPublishedCount, CreatedAt
        )
        OUTPUT INSERTED.PublicationBatchID, INSERTED.PublishedAt
        VALUES (?, GETDATE(), ?, ?, GETDATE())
        """,
        next_serial, published_by_user_id, cases_published_count
    )
    row = cursor.fetchone()

    return {
        "publication_batch_id": row[0],
        "publication_serial": next_serial,
        "published_at": row[1],
    }


def create_publication_batch_case(
    cursor,
    publication_batch_id: int,
    incident_case_id: int,
    administrative_subcase_id: Optional[int] = None,
    target_org_unit_id: Optional[int] = None
) -> None:
    """
    Insert one APP_PublicationBatchCase row using the caller's open cursor/transaction.
    AdministrativeSubcaseID / TargetOrgUnitID are stored as NULL when not available.
    """
    cursor.execute(
        """
        INSERT INTO dbo.APP_PublicationBatchCase (
            PublicationBatchID, IncidentCaseID, AdministrativeSubcaseID, TargetOrgUnitID, CreatedAt
        )
        VALUES (?, ?, ?, ?, GETDATE())
        """,
        publication_batch_id, incident_case_id, administrative_subcase_id, target_org_unit_id
    )


def get_recent_publication_batches(limit: int = 5) -> List[Dict[str, Any]]:
    """
    Return the most recent publication batches, newest first.

    Joins APP_Users to resolve a display name for PublishedByUserID
    (DisplayName, falling back to Username).
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT TOP (?)
                pb.PublicationBatchID,
                pb.PublicationSerial,
                pb.PublishedAt,
                pb.CasesPublishedCount,
                u.Username AS PublishedByUsername,
                u.DisplayName AS PublishedByDisplayName
            FROM dbo.APP_PublicationBatch pb
            LEFT JOIN dbo.APP_Users u ON pb.PublishedByUserID = u.UserID
            ORDER BY pb.PublicationBatchID DESC
            """,
            limit
        )
        rows = cursor.fetchall()

        result = []
        for row in rows:
            published_by = row.PublishedByDisplayName or row.PublishedByUsername
            result.append({
                "publication_batch_id": row.PublicationBatchID,
                "publication_serial": row.PublicationSerial,
                "published_at": row.PublishedAt,
                "published_by": published_by,
                "cases_published_count": row.CasesPublishedCount,
            })

        return result

    finally:
        cursor.close()
        conn.close()


def get_publication_batch_cases(publication_batch_id: int) -> List[Dict[str, Any]]:
    """
    Return the cases included in a single publication batch.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            SELECT
                PublicationBatchCaseID,
                IncidentCaseID,
                AdministrativeSubcaseID,
                TargetOrgUnitID
            FROM dbo.APP_PublicationBatchCase
            WHERE PublicationBatchID = ?
            ORDER BY PublicationBatchCaseID
            """,
            publication_batch_id
        )
        rows = cursor.fetchall()

        return [
            {
                "publication_batch_case_id": row.PublicationBatchCaseID,
                "incident_case_id": row.IncidentCaseID,
                "administrative_subcase_id": row.AdministrativeSubcaseID,
                "target_org_unit_id": row.TargetOrgUnitID,
            }
            for row in rows
        ]

    finally:
        cursor.close()
        conn.close()
