"""
Import Router — Hospital Data Intake Pipeline
GET  /api/import/template            — download the Excel import template
POST /api/import/upload              — upload filled template, get a review preview (nothing committed yet)
POST /api/import/{batch_id}/confirm  — commit a previously staged upload after review
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse

from ..services import import_service

router = APIRouter(prefix="/api/import", tags=["Import"])


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
