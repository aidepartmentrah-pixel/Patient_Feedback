"""
ML Embedding Processing Worker

Stage 6 of the ML architecture consolidation (see ML_ARCHITECTURE_DECISION_RECORD.md
and the approved execution plan). Processes ml.EmbeddingProcessingJob rows
asynchronously and in batches, replacing the old synchronous, unbatched,
silently-swallowed-failure embedding calls in ml_insert_adapter.py.

Runs as a background thread INSIDE the existing FastAPI process — the same
pattern already proven by training_service.run_training_pipeline() — not a
separate process. This means it reuses whatever embedding model instance
startup_ml_warmup() already loaded into this process's memory (module-level
_TOKENIZER/_MODEL globals in modular_functions.py), rather than loading a
second copy.

Only the two confirmed load-bearing embeddings are computed
(embedding_text1 / embedding_text123 — see
ML_ARCHITECTURE_DECISION_RECORD.md section 4.5), via the batch-capable
get_embedding_list() rather than the one-at-a-time get_embedding() the old
adapter used.
"""

import threading
import traceback
from typing import Dict

from core.database import get_connection
from api.db_layer import ml_embedding_job_db, ml_case_training_db, ml_model_version_db

_WORKER_THREAD = None
_STOP_EVENT = threading.Event()

BATCH_SIZE = 10
POLL_INTERVAL_SECONDS = 10
STUCK_JOB_THRESHOLD_MINUTES = 15

# The code calls this model "MPNet"; its own saved config identifies it as
# XLMRobertaModel (see ML_ARCHITECTURE_DECISION_RECORD.md — this is exactly
# why ml.EmbeddingModelVersion exists: never trust a label alone).
MODEL_NAME = "mpnet_embeddings (local)"
MODEL_PATH = "models_directory/Classification_Models/model_storage/mpnet_embeddings"
MODEL_ARCHITECTURE = "XLMRobertaModel"
EMBEDDING_DIMENSION = 768


def _get_embedding_list_fn():
    """Lazy import — avoids pulling in torch/transformers at module import time."""
    from models_directory.Classification_Models.Stage.modular_functions import get_embedding_list
    return get_embedding_list


def _ensure_active_model_version(cursor) -> int:
    """Registers the model version once; reuses it on subsequent calls."""
    existing = ml_model_version_db.get_active_model_version(cursor)
    if existing:
        return existing["EmbeddingModelVersionID"]
    return ml_model_version_db.register_model_version(
        cursor,
        model_name=MODEL_NAME,
        model_path_or_identifier=MODEL_PATH,
        embedding_dimension=EMBEDDING_DIMENSION,
        model_architecture=MODEL_ARCHITECTURE,
        pooling_method="mean",
        tokenizer_identifier=MODEL_PATH,
    )


def process_pending_jobs(batch_size: int = BATCH_SIZE) -> Dict[str, int]:
    """
    Claim up to batch_size Pending/RetryPending jobs, batch-generate
    embeddings for each distinct case among them (one case may have several
    queued jobs — e.g. Create followed by an edit before the worker got to
    it — so embeddings are computed once per case, and every job for that
    case is marked together), and mark jobs Completed/Failed.

    Returns {"claimed": n, "completed": n, "failed": n}.
    """
    get_embedding_list = _get_embedding_list_fn()

    conn = get_connection()
    cursor = conn.cursor()
    try:
        jobs = ml_embedding_job_db.claim_pending_jobs(cursor, batch_size)
        conn.commit()  # release the 'Processing' status promptly, before the (possibly slow) model call
    finally:
        cursor.close()
        conn.close()

    if not jobs:
        return {"claimed": 0, "completed": 0, "failed": 0}

    case_ids = sorted({j["incident_request_case_id"] for j in jobs})

    conn = get_connection()
    cursor = conn.cursor()
    completed = 0
    failed = 0

    try:
        model_version_id = _ensure_active_model_version(cursor)
        conn.commit()

        complaint_texts = []
        combined_texts = []
        valid_case_ids = []
        for case_id in case_ids:
            record = ml_case_training_db.get_case_training_record(cursor, case_id)
            if record is None:
                continue
            complaint_text = record.get("ComplaintText") or ""
            immediate = record.get("ImmediateActionText") or ""
            taken = record.get("TakenActionText") or ""
            complaint_texts.append(complaint_text)
            combined_texts.append(f"{complaint_text} {immediate} {taken}".strip())
            valid_case_ids.append(case_id)

        complaint_embeddings = get_embedding_list(complaint_texts) if complaint_texts else []
        combined_embeddings = get_embedding_list(combined_texts) if combined_texts else []

        embeddings_by_case = {
            case_id: (complaint_embeddings[i], combined_embeddings[i])
            for i, case_id in enumerate(valid_case_ids)
        }

        for job in jobs:
            case_id = job["incident_request_case_id"]
            job_id = job["job_id"]
            try:
                if case_id not in embeddings_by_case:
                    raise ValueError(f"No ml.CaseTrainingRecord found for case {case_id}")
                complaint_emb, combined_emb = embeddings_by_case[case_id]
                ml_case_training_db.update_embeddings(
                    cursor, case_id, complaint_emb, combined_emb, model_version_id, EMBEDDING_DIMENSION
                )
                ml_embedding_job_db.mark_job_completed(cursor, job_id, model_version_id)
                completed += 1
            except Exception as e:
                ml_embedding_job_db.mark_job_failed(cursor, job_id, error_code=type(e).__name__, error_message=str(e))
                failed += 1

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

    return {"claimed": len(jobs), "completed": completed, "failed": failed}


def sweep_stuck_jobs_startup(stuck_threshold_minutes: int = STUCK_JOB_THRESHOLD_MINUTES) -> int:
    """
    Call once at process startup: recover jobs left in 'Processing' from a
    prior crash/restart (the job table, not in-memory state, is always the
    source of truth — see ML_ARCHITECTURE_DECISION_RECORD.md).
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        recovered = ml_embedding_job_db.sweep_stuck_jobs(cursor, stuck_threshold_minutes)
        conn.commit()
        return recovered
    finally:
        cursor.close()
        conn.close()


def _worker_loop(poll_interval_seconds: int):
    while not _STOP_EVENT.is_set():
        try:
            result = process_pending_jobs()
            if result["claimed"] > 0:
                print(f"[ML WORKER] Processed batch: {result}")
        except Exception as e:
            print(f"[ML WORKER ERROR] {e}")
            traceback.print_exc()
        _STOP_EVENT.wait(poll_interval_seconds)


def start_worker_background_thread(poll_interval_seconds: int = POLL_INTERVAL_SECONDS):
    """
    Idempotent — safe to call more than once; only ever starts one worker
    thread per process. Intended to be called from a FastAPI startup event
    (see backend/main.py), the same way training/STT model warmup already is.
    """
    global _WORKER_THREAD
    if _WORKER_THREAD is not None and _WORKER_THREAD.is_alive():
        return

    recovered = sweep_stuck_jobs_startup()
    if recovered:
        print(f"[ML WORKER] Recovered {recovered} job(s) stuck in Processing from a prior run")

    _STOP_EVENT.clear()
    _WORKER_THREAD = threading.Thread(target=_worker_loop, args=(poll_interval_seconds,), daemon=True)
    _WORKER_THREAD.start()
    print("[ML WORKER] Background worker thread started")


def stop_worker_background_thread():
    """For clean shutdown in tests; not required in production (daemon thread)."""
    _STOP_EVENT.set()
