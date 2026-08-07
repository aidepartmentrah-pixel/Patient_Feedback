"""
Import Router — Hospital Data Intake Pipeline
GET    /api/import/template               — download the Excel import template
POST   /api/import/upload                 — upload filled template, get a review preview (nothing committed yet)
POST   /api/import/{batch_id}/confirm     — commit a previously staged upload after review
GET    /api/import/batches                — batch history (list of past uploads)
GET    /api/import/{batch_id}/report      — re-download a past batch's report
GET    /api/import/{batch_id}/preview     — resume the review screen for a still-PendingReview batch
GET    /api/import/lookups                — plain lookup lists for in-grid editing (doctor/worker use live search instead)
PATCH  /api/import/{batch_id}/rows/{n}    — edit one row's fields in the staged file
DELETE /api/import/{batch_id}             — discard a still-PendingReview batch
"""

from typing import Any, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from ..services import import_service

router = APIRouter(prefix="/api/import", tags=["Import"])


class RowPatchRequest(BaseModel):
    fields: Dict[str, Any]


@router.get("/template")
async def download_template():
    """
    Download the Excel import template pre-loaded with live DB dropdowns.
    """
    try:
        buf = import_service.generate_template()
        return StreamingResponse(
            buf,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=import_template.xlsx"},
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Template generation failed: {exc}")


@router.post("/upload")
async def upload_import_file(file: UploadFile = File(...)):
    """
    Upload a filled import template (.xlsx) for review.
    Parses, validates, and groups by incident, but writes nothing to the
    database yet. Returns a preview (one entry per incident group, colored
    green/yellow/red/duplicate) plus an import_batch_id to pass to
    POST /{batch_id}/confirm once the user has reviewed it.
    """
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        preview = import_service.stage_upload(contents, uploaded_by_user_id=1)
        return JSONResponse(content=preview)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import processing failed: {exc}")


@router.post("/{batch_id}/confirm")
async def confirm_import_batch(batch_id: int):
    """
    Commit a previously staged upload (see POST /upload) after the user has
    reviewed the preview. Re-validates the staged file fresh right before
    committing, then imports every fully valid incident group.
    """
    try:
        report = import_service.confirm_import(batch_id, confirmed_by_user_id=1)
        return JSONResponse(content=report)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import confirmation failed: {exc}")


@router.get("/{batch_id}/preview")
async def resume_batch_preview(batch_id: int):
    """
    Resume the review screen for a batch that's still PendingReview (e.g.
    after a page refresh lost the in-browser preview state). Re-validates
    the still-staged file fresh; does not create a new batch or commit anything.
    """
    try:
        preview = import_service.resume_preview(batch_id)
        return JSONResponse(content=preview)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to resume review: {exc}")


@router.get("/batches")
async def get_import_batches(limit: int = 50):
    """Batch history: past uploads with their outcome counts, newest first."""
    try:
        return JSONResponse(content=import_service.list_import_batches(limit=limit))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load batch history: {exc}")


@router.get("/{batch_id}/report")
async def download_batch_report(batch_id: int):
    """Re-download a past batch's report Excel."""
    path = import_service.get_report_path(batch_id)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No report available for batch {batch_id}.")
    return StreamingResponse(
        open(path, "rb"),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=import_report_{batch_id}.xlsx"},
    )


@router.get("/lookups")
async def get_editable_lookups():
    """
    Plain internal lookup lists (Classification, Department, Domain, etc.)
    for the review grid's inline editors -- fetch once, filter client-side.
    Doctor/Worker aren't included here; they use the existing live search
    endpoints (GET /api/records/search/doctors|employees) instead.
    """
    try:
        return JSONResponse(content=import_service.get_editable_lookups())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load lookups: {exc}")


@router.patch("/{batch_id}/rows/{row_number}")
async def patch_row(batch_id: int, row_number: int, body: RowPatchRequest):
    """Edit one row's field(s) directly in the staged file, then re-validate."""
    try:
        preview = import_service.patch_staged_rows(batch_id, [{"row_number": row_number, "fields": body.fields}])
        return JSONResponse(content=preview)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to apply edit: {exc}")


@router.delete("/{batch_id}")
async def delete_batch(batch_id: int):
    """Discard a still-PendingReview batch (its staged file and DB row)."""
    try:
        import_service.discard_batch(batch_id)
        return JSONResponse(content={"success": True})
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to discard batch: {exc}")
