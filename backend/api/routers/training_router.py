"""
Training Router
API endpoints for the Settings > Training Tab
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List

from ..services.training_service import (
    run_training_pipeline,
    get_training_status,
    get_training_history_list,
    get_ml_database_size_history,
    is_training_running
)


router = APIRouter(prefix="/api/settings/training", tags=["Settings - Training"])


# ==================== ENDPOINTS ====================

@router.get("/status")
async def get_training_status_endpoint() -> Dict[str, Any]:
    """
    Get current model performance metrics.
    
    **Returns:**
    - last_run: ISO timestamp of last training run
    - status: "never_run" | "running" | "completed" | "failed"
    - models: List of model performance metrics
    
    **Example Response:**
    ```json
    {
      "last_run": "2026-01-02T11:43:00",
      "status": "completed",
      "models": [
        {
          "model_name": "Domain_Model",
          "num_records": 412,
          "accuracy": 0.8234,
          "precision": 0.8011,
          "recall": 0.7988,
          "f1": 0.7999,
          "last_trained": "2026-01-02T11:41:00"
        }
      ]
    }
    ```
    """
    try:
        return get_training_status()
    except Exception as e:
        print(f"[TRAINING ERROR] get_training_status_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training status")


@router.get("/history")
async def get_training_history_endpoint() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get historical training runs.
    
    **Returns:**
    - history: List of training run records
    
    **Example Response:**
    ```json
    {
      "history": [
        {
          "run_id": "2026_01_02_1140",
          "started_at": "2026-01-02T11:40:00",
          "finished_at": "2026-01-02T11:43:00",
          "status": "completed",
          "models_trained": 12
        }
      ]
    }
    ```
    """
    try:
        return get_training_history_list()
    except Exception as e:
        print(f"[TRAINING ERROR] get_training_history_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training history")


@router.get("/db-size")
async def get_ml_database_size_endpoint() -> Dict[str, List[Dict[str, Any]]]:
    """
    Get ML database size growth over time for graphing.
    
    **Returns:**
    - points: List of {date, records} points
    
    **Example Response:**
    ```json
    {
      "points": [
        { "date": "2026-01-01", "records": 153 },
        { "date": "2026-01-02", "records": 191 },
        { "date": "2026-01-03", "records": 248 }
      ]
    }
    ```
    """
    try:
        return get_ml_database_size_history()
    except Exception as e:
        print(f"[TRAINING ERROR] get_ml_database_size_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database size history")


@router.post("/run")
async def trigger_training_endpoint() -> Dict[str, Any]:
    """
    Trigger full model retraining pipeline.
    
    **Behavior:**
    - Runs asynchronously (non-blocking)
    - Returns immediately with run_id
    - Training continues in background
    - Results stored after completion
    
    **Returns:**
    - status: "started"
    - run_id: Unique run identifier (format: YYYY_MM_DD_HHMM)
    
    **Example Response:**
    ```json
    {
      "status": "started",
      "run_id": "2026_01_05_1430"
    }
    ```
    
    **Notes:**
    - Check GET /status to monitor progress
    - Multiple concurrent runs are prevented
    """
    try:
        # Check if training already running
        if is_training_running():
            raise HTTPException(
                status_code=409,
                detail="Training already in progress. Please wait for completion."
            )
        
        # Start training pipeline
        result = run_training_pipeline()
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[TRAINING ERROR] trigger_training_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start training")
