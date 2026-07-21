"""
Training Router
API endpoints for the Settings > Training Tab
"""

from fastapi import APIRouter, HTTPException, Depends, Response
from typing import Dict, Any, List

from ..dependencies.user_context import get_current_user
from ..schemas.auth_models import CurrentUser
from ..utils.guards import require_logged_in, require_role
from core.constants.roles import SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR
from ..services.training_service import (
    run_training_pipeline,
    get_training_status,
    get_training_history_list,
    get_ml_database_size_history,
    is_training_running,
    get_training_progress,
    get_grouped_training_status,
    get_db_growth_chart_data,
    get_performance_trends_chart_data,
    get_training_timeline_chart_data,
    get_family_comparison_chart_data
)
from ..services.training_run_artifacts_service import (
    list_versioned_runs,
    get_versioned_run_detail,
    build_run_zip,
    RunNotFoundError,
)


router = APIRouter(prefix="/api/settings/training", tags=["Settings - Training"])


# ==================== ENDPOINTS ====================

@router.get("/status")
async def get_training_status_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
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
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        return get_training_status()
    except Exception as e:
        print(f"[TRAINING ERROR] get_training_status_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training status")


@router.get("/progress")
async def get_training_progress_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get real-time training progress (for polling during training).
    
    **Returns:**
    - is_running: Whether training is currently in progress
    - run_id: Current run identifier
    - current_model: Name of model being trained
    - current_step: Current step number
    - total_steps: Total number of training steps
    - progress_percentage: Progress as percentage (0-100)
    - elapsed_seconds: Time elapsed since training started
    - estimated_remaining_seconds: Estimated time remaining
    - last_completed: Last completed model name
    
    **Example Response (Training in Progress):**
    ```json
    {
      "is_running": true,
      "run_id": "2026_01_21_1430",
      "current_model": "Severity_Model",
      "current_step": 11,
      "total_steps": 18,
      "progress_percentage": 61,
      "elapsed_seconds": 58,
      "estimated_remaining_seconds": 37,
      "last_completed": "Harm_Ordinal_Low"
    }
    ```
    
    **Example Response (Not Running):**
    ```json
    {
      "is_running": false,
      "run_id": null,
      "current_model": null,
      "current_step": 0,
      "total_steps": 18,
      "progress_percentage": 0,
      "elapsed_seconds": 0,
      "estimated_remaining_seconds": 0,
      "last_completed": null
    }
    ```
    
    **Usage:**
    - Poll this endpoint every 2-3 seconds while training
    - Stop polling when is_running becomes false
    - Use progress_percentage for progress bar
    - Display current_model and estimated_remaining_seconds to user
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        return get_training_progress()
    except Exception as e:
        print(f"[TRAINING ERROR] get_training_progress_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training progress")


@router.get("/grouped-status")
async def get_grouped_training_status_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get training status with models grouped by family and performance alerts.
    
    **Returns:**
    - last_run: ISO timestamp of last training run
    - status: Training status
    - model_families: List of model families with aggregated metrics
    - alerts: List of performance alerts
    - summary: Overall summary statistics
    
    **Example Response:**
    ```json
    {
      "last_run": "2026-01-21T13:48:35",
      "status": "completed",
      "model_families": [
        {
          "family_key": "hierarchical",
          "family_name": "Hierarchical Models",
          "family_name_ar": "نماذج التسلسل الهرمي",
          "model_count": 11,
          "avg_f1": 0.6820,
          "avg_accuracy": 0.7180,
          "avg_precision": 0.6940,
          "avg_recall": 0.7020,
          "total_records": 371,
          "models": [
            {
              "model_name": "Domain_Model",
              "num_records": 371,
              "accuracy": 0.7530,
              "precision": 0.7370,
              "recall": 0.7530,
              "f1": 0.7310,
              "last_trained": "2026-01-21T13:48:35"
            }
          ]
        },
        {
          "family_key": "harm_assessment",
          "family_name": "Harm Assessment",
          "family_name_ar": "تقييم الأضرار",
          "model_count": 3,
          "avg_f1": 0.4933,
          "avg_accuracy": 0.5963,
          "total_records": 740,
          "models": [...]
        }
      ],
      "alerts": [
        {
          "severity": "critical",
          "model_name": "Classification_En",
          "metric": "f1_score",
          "value": 0.141,
          "message": "Classification_En has critically low F1 score (14.1%)",
          "message_ar": "Classification_En لديه درجة F1 منخفضة للغاية (14.1%)",
          "recommendation": "Review training data quality and model architecture",
          "recommendation_ar": "مراجعة جودة بيانات التدريب وبنية النموذج"
        },
        {
          "severity": "warning",
          "model_name": "Subcategory_Cat1",
          "metric": "training_data",
          "value": 0,
          "message": "Subcategory_Cat1 has no training data - model could not train",
          "message_ar": "Subcategory_Cat1 ليس لديه بيانات تدريب",
          "recommendation": "Add data for this category to enable training",
          "recommendation_ar": "إضافة بيانات لهذه الفئة لتمكين التدريب"
        }
      ],
      "summary": {
        "total_models": 18,
        "total_families": 4,
        "overall_avg_f1": 0.7234,
        "critical_alerts": 1,
        "warning_alerts": 5
      }
    }
    ```
    
    **Features:**
    - Models grouped by logical families (Hierarchical, Harm, Classification, Severity)
    - Aggregated metrics per family (average F1, accuracy, precision, recall)
    - Performance alerts for problematic models
    - Summary statistics for quick overview
    - Bilingual support (English/Arabic)
    
    **Use Cases:**
    - Display organized model performance dashboard
    - Show performance alerts to users
    - Identify models needing attention
    - Compare family-level performance
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        return get_grouped_training_status()
    except Exception as e:
        print(f"[TRAINING ERROR] get_grouped_training_status_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve grouped training status")


@router.get("/history")
async def get_training_history_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
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
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        return get_training_history_list()
    except Exception as e:
        print(f"[TRAINING ERROR] get_training_history_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training history")


@router.get("/versioned-runs")
async def list_versioned_runs_endpoint(
    limit: int = 50,
    current_user: CurrentUser = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """
    List Stage 9 versioned training runs (per-run folders with ROC/AUC
    artifacts) — separate from GET /history, which is the older SQLite-
    backed run log.

    **Returns:** list of {run_id, status, started_at, finished_at,
    model_count, families_with_warnings}, newest first.
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])

    try:
        return list_versioned_runs(limit=limit)
    except Exception as e:
        print(f"[TRAINING ERROR] list_versioned_runs_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list versioned training runs")


@router.get("/versioned-runs/{run_id}")
async def get_versioned_run_detail_endpoint(
    run_id: str,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """Full run_summary.json for one versioned run, including the
    per-model artifact manifest."""
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])

    try:
        return get_versioned_run_detail(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[TRAINING ERROR] get_versioned_run_detail_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve training run detail")


@router.get("/versioned-runs/{run_id}/download")
async def download_versioned_run_endpoint(
    run_id: str,
    current_user: CurrentUser = Depends(get_current_user)
):
    """Download a ZIP of every artifact in a versioned run's manifest
    (confusion matrices, ROC/PR curves, model files, run_summary.json)."""
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])

    try:
        result = build_run_zip(run_id)
    except RunNotFoundError:
        raise HTTPException(status_code=404, detail=f"Training run not found: {run_id}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"[TRAINING ERROR] download_versioned_run_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to build training run ZIP")

    return Response(
        content=result["content"],
        media_type=result["content_type"],
        headers={"Content-Disposition": f'attachment; filename="{result["filename"]}"'},
    )


@router.get("/db-size")
async def get_ml_database_size_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, List[Dict[str, Any]]]:
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
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    try:
        return get_ml_database_size_history()
    except Exception as e:
        print(f"[TRAINING ERROR] get_ml_database_size_endpoint failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database size history")


@router.post("/run")
async def trigger_training_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
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
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
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


# ==================== CHART ENDPOINTS (PHASE 4) ====================

@router.get("/charts/db-growth")
async def get_db_growth_chart_endpoint(
    days: int = 30,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get ML database growth chart data (Phase 4).
    
    Chart-ready data formatted for Chart.js/Recharts visualization:
    - Area chart showing database growth over time
    - Growth percentage and total records
    - Configurable time range
    
    Query Parameters:
        - days: Number of days to include (default: 30)
    
    Returns:
        - labels: Array of dates
        - datasets: Chart datasets with styling
        - metadata: Growth statistics and date range
    
    Example Response:
        {
            "labels": ["2026-01-01", "2026-01-02", ...],
            "datasets": [{
                "label": "Database Records",
                "data": [100, 150, 200, ...],
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
                "borderColor": "rgba(54, 162, 235, 1)",
                ...
            }],
            "metadata": {
                "total_points": 30,
                "growth": {"total": 100, "percentage": 50.0},
                ...
            }
        }
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    return get_db_growth_chart_data(days=days)


@router.get("/charts/performance-trends")
async def get_performance_trends_chart_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get model performance trends chart data (Phase 4).
    
    Multi-line chart showing F1 score trends over last 10 training runs:
    - One line per model family
    - Tracks performance improvements/degradations
    - Bilingual labels (English/Arabic)
    
    Returns:
        - labels: Array of run IDs
        - datasets: One dataset per model family with F1 scores
        - metadata: Total runs and family count
    
    Example Response:
        {
            "labels": ["2026_01_15_1200", "2026_01_16_0900", ...],
            "datasets": [
                {
                    "label": "Hierarchical Models",
                    "label_ar": "نماذج التسلسل الهرمي",
                    "data": [0.65, 0.67, 0.69, ...],
                    "borderColor": "rgba(75, 192, 192, 1)",
                    ...
                },
                ...
            ],
            "metadata": {"total_runs": 10, "families": 4}
        }
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    return get_performance_trends_chart_data()


@router.get("/charts/training-timeline")
async def get_training_timeline_chart_endpoint(
    limit: int = 20,
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get training timeline chart data (Phase 4).
    
    Bar chart showing training run durations and status:
    - Duration in seconds for each run
    - Color-coded by status (green=completed, red=failed)
    - Success rate and average duration metrics
    
    Query Parameters:
        - limit: Maximum number of runs to show (default: 20)
    
    Returns:
        - labels: Array of run IDs
        - datasets: Training durations with status colors
        - metadata: Success rate, average duration, run counts
    
    Example Response:
        {
            "labels": ["2026_01_15_1200", "2026_01_16_0900", ...],
            "datasets": [{
                "label": "Training Duration (seconds)",
                "data": [120, 115, 118, ...],
                "backgroundColor": ["rgba(75, 192, 192, 0.6)", ...],
                ...
            }],
            "metadata": {
                "avg_duration": 117.5,
                "success_rate": 95.0,
                "successful_runs": 19,
                "failed_runs": 1
            }
        }
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    return get_training_timeline_chart_data(limit=limit)


@router.get("/charts/family-comparison")
async def get_family_comparison_chart_endpoint(
    current_user: CurrentUser = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get family comparison chart data (Phase 4).
    
    Grouped bar chart comparing current performance across model families:
    - Shows F1 Score, Accuracy, Precision, Recall for each family
    - Side-by-side comparison of all metrics
    - Bilingual labels (English/Arabic)
    
    Returns:
        - labels: Array of family names
        - labels_ar: Array of Arabic family names
        - datasets: Four datasets (one per metric)
        - metadata: Family count and record counts
    
    Example Response:
        {
            "labels": ["Hierarchical Models", "Harm Assessment", ...],
            "labels_ar": ["نماذج التسلسل الهرمي", "تقييم الأضرار", ...],
            "datasets": [
                {
                    "label": "F1 Score",
                    "data": [0.62, 0.49, 0.71, 0.72],
                    "backgroundColor": "rgba(54, 162, 235, 0.6)",
                    ...
                },
                ...
            ],
            "metadata": {
                "total_families": 4,
                "record_counts": [1084, 740, 1215, 365]
            }
        }
    """
    require_logged_in(current_user)
    require_role(current_user, [SOFTWARE_ADMIN, COMPLAINT_SUPERVISOR])
    
    return get_family_comparison_chart_data()
