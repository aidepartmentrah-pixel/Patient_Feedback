"""
ML Embedding Model Version — DB Layer
Pure SQL operations only. No business logic.

Never trust a model label alone (the existing code calls the live embedding
model "MPNet" while its own saved config identifies it as XLMRobertaModel) —
this table lets every stored embedding be traced to the exact model/version/
config that produced it.
"""

from typing import Any, Dict, Optional


def get_active_model_version(cursor) -> Optional[Dict[str, Any]]:
    cursor.execute(
        "SELECT TOP 1 * FROM ml.EmbeddingModelVersion WHERE IsActive = 1 ORDER BY ActivatedAt DESC"
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [c[0] for c in cursor.description]
    return dict(zip(columns, row))


def register_model_version(
    cursor,
    model_name: str,
    model_path_or_identifier: str,
    embedding_dimension: int,
    model_architecture: Optional[str] = None,
    model_checksum: Optional[str] = None,
    pooling_method: Optional[str] = None,
    normalization_method: Optional[str] = None,
    tokenizer_identifier: Optional[str] = None,
    configuration_json: Optional[str] = None,
    deactivate_others: bool = True,
) -> int:
    """
    Register a new model version as the active one. By default, retires
    (IsActive=0, RetiredAt=now) any previously active version first, so
    there is always at most one active model version at a time.
    """
    if deactivate_others:
        cursor.execute(
            "UPDATE ml.EmbeddingModelVersion SET IsActive = 0, RetiredAt = GETDATE() WHERE IsActive = 1"
        )

    cursor.execute(
        """
        INSERT INTO ml.EmbeddingModelVersion (
            ModelName, ModelPathOrIdentifier, ModelArchitecture, ModelChecksum,
            EmbeddingDimension, PoolingMethod, NormalizationMethod, TokenizerIdentifier,
            ConfigurationJson, IsActive
        )
        OUTPUT INSERTED.EmbeddingModelVersionID
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (model_name, model_path_or_identifier, model_architecture, model_checksum,
         embedding_dimension, pooling_method, normalization_method, tokenizer_identifier,
         configuration_json),
    )
    return cursor.fetchone()[0]
