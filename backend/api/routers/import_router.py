"""
Import Router — Hospital Data Intake Pipeline
GET  /api/import/template   — download the Excel import template
POST /api/import/upload     — upload filled template, get import report
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
    Upload a filled import template (.xlsx).
    Returns a structured import report. Rejected rows are base64-encoded Excel
    in the response field 'rejected_excel_b64'.
    """
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are accepted.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        report = import_service.process_upload(contents, created_by_user_id=1)
        return JSONResponse(content=report)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Import processing failed: {exc}")
