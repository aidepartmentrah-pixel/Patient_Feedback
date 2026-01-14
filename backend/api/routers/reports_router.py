"""
Reports Router
FastAPI endpoints for the Reporting Page.
Handles seasonal reports and report exports.
"""

# Standard library imports
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Literal
import uuid
import traceback
from io import BytesIO

# FastAPI imports
from fastapi import APIRouter, Query, HTTPException, Response
from pydantic import BaseModel, Field

# Import for emergency fallback
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

# Service imports
from ..services.reports_service import reports_service
from backend.api.services.monthly_report_service import monthly_report_service
from backend.api.services.report_export_service import report_export_service
from backend.api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
from backend.api.services.seasonal_report_explanation_service import SeasonalReportExplanationService



# ============================================================
# ROUTER CONFIGURATION
# ============================================================

router = APIRouter(prefix="/api/reports", tags=["Reports"])

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class ExportRequest(BaseModel):
    """Request model for exporting reports."""
    report_type: Literal["monthly", "seasonal"]
    display_mode: Literal["detailed", "numeric", "hcat"] = "detailed"
    year: int
    month: Optional[int] = None
    trimester: Optional[int] = None
    quarter: Optional[int] = None
    filters: Optional[Dict[str, Any]] = {}
    include_charts: bool = True
    include_metadata: bool = True
    language: Literal["en", "ar"] = "en"


class SeasonalViewRequest(BaseModel):
    """Request model for viewing seasonal reports."""
    season_id: int
    orgunit_id: int
    user_id: int


class SeasonalViewRequestV2(BaseModel):
    """Request model for viewing seasonal reports (V2 - year/trimester based)."""
    year: int
    trimester: str  # Trim1, Trim2, Trim3
    orgunit_id: int
    orgunit_type: int
    user_id: Optional[int] = 1


class MonthlyViewRequest(BaseModel):
    """Request model for viewing monthly reports."""
    year: int
    month: Optional[int] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    mode: Literal["detailed", "numeric"] = "detailed"
    scope: Optional[str] = None
    administration_ids: Optional[str] = None
    department_ids: Optional[str] = None
    section_ids: Optional[str] = None


class SubmitExplanationRequest(BaseModel):
    """Request body for submitting/updating an explanation for a seasonal report."""
    explanation_text: str = Field(..., description="Explanation text from the organizational unit")
    submitted_by_user_id: int = Field(..., description="User ID who submitted the explanation")

    class Config:
        json_schema_extra = {
            "example": {
                "explanation_text": "The increase in incidents was due to staffing changes during the season...",
                "submitted_by_user_id": 1
            }
        }


class ExplanationResponse(BaseModel):
    """Response after submitting or updating an explanation."""
    status: str = Field(..., description="Operation status")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok"
            }
        }


# In-memory storage for exports
EXPORT_STORAGE: Dict[str, Dict[str, Any]] = {}

# ============================================================
# SEASONAL REPORT ENDPOINTS
# ============================================================

@router.post("/seasonal/view", response_model=Dict[str, Any])
def view_seasonal_report(request: SeasonalViewRequestV2):
    """View a seasonal report (generates if needed)."""
    from backend.api.db_layer.seasonal_report import resolve_season_id_from_year_trimester
    
    # Resolve season_id from year + trimester
    try:
        season_id = resolve_season_id_from_year_trimester(
            year=request.year,
            trimester=request.trimester
        )
    except ValueError as e:
        # Ambiguous season (multiple matches)
        error_msg = str(e)
        if "Ambiguous" in error_msg:
            raise HTTPException(status_code=409, detail=error_msg)
        elif "Invalid trimester" in error_msg:
            raise HTTPException(status_code=400, detail=error_msg)
        else:
            raise HTTPException(status_code=400, detail=str(e))
    
    if season_id is None:
        raise HTTPException(
            status_code=404,
            detail=f"Season not found for year={request.year}, trimester={request.trimester}"
        )
    
    try:
        report = get_or_generate_seasonal_report(
            season_id=season_id,
            orgunit_id=request.orgunit_id,
            orgunit_type=request.orgunit_type,
            user_id=request.user_id
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/monthly/view", response_model=Dict[str, Any])
def view_monthly_report(request: MonthlyViewRequest):
    """View a monthly report."""
    try:
        result = monthly_report_service.generate_monthly_report(
            year=request.year,
            month=request.month,
            start_date=request.start_date,
            end_date=request.end_date,
            mode=request.mode,
            scope=request.scope,
            administration_ids=request.administration_ids,
            department_ids=request.department_ids,
            section_ids=request.section_ids,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/seasonal/{report_id}/explanation", response_model=ExplanationResponse)
def submit_explanation(report_id: int, request: SubmitExplanationRequest):
    """Submit an explanation for a seasonal report."""
    try:
        explanation_service = SeasonalReportExplanationService()
        explanation_service.submit_explanation(
            seasonal_report_id=report_id,
            explanation_text=request.explanation_text,
            submitted_by_user_id=request.submitted_by_user_id
        )
        return ExplanationResponse(status="ok")
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail=str(e))
        elif "already exists" in error_msg or "duplicate" in error_msg:
            raise HTTPException(status_code=409, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit explanation: {str(e)}")


@router.put("/seasonal/{report_id}/explanation", response_model=ExplanationResponse)
def update_explanation(report_id: int, request: SubmitExplanationRequest):
    """Update an existing explanation for a seasonal report."""
    try:
        explanation_service = SeasonalReportExplanationService()
        explanation_service.update_explanation(
            seasonal_report_id=report_id,
            explanation_text=request.explanation_text,
            submitted_by_user_id=request.submitted_by_user_id
        )
        return ExplanationResponse(status="ok")
    except ValueError as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update explanation: {str(e)}")


# ============================================================
# EXPORT ENDPOINTS
# ============================================================

@router.post("/export")
async def export_report(request: ExportRequest, format: Literal["pdf", "csv", "xlsx", "docx"] = Query(..., description="Export format")):
    """
    Export a report in the specified format.
    
    Unified endpoint for all export formats (PDF, CSV, Excel, Word).
    """
    try:
        result = report_export_service.generate_export(
            report_type=request.report_type,
            display_mode=request.display_mode,
            file_format=format,
            year=request.year,
            month=request.month,
            trimester=request.trimester,
            quarter=request.quarter,
            filters=request.filters,
            include_charts=request.include_charts,
            language=request.language
        )
        
        export_id = f"exp-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        EXPORT_STORAGE[export_id] = {
            "filename": result["filename"],
            "content": result["content"],
            "content_type": result["content_type"],
            "created_at": datetime.now(),
            "user_id": None,
            "filters_applied": request.filters
        }
        
        return {
            "export_id": export_id,
            "file_name": result["filename"],
            "file_size_bytes": len(result["content"]),
            "download_url": f"/api/reports/download/{export_id}",
            "generated_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(hours=24)).isoformat(),
            "audit_logged": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": f"{format}_export_failed",
            "message": f"Failed to generate {format.upper()}: {str(e)}",
            "message_ar": f"فشل إنشاء ملف {format.upper()}: {str(e)}"
        })


@router.post("/monthly/export")
async def export_monthly_report(
    year: int = Query(..., description="Year for the report"),
    month: int = Query(..., description="Month for the report"),
    format: Literal["pdf", "csv", "xlsx", "docx"] = Query(..., description="Export format"),
    display_mode: Literal["detailed", "numeric"] = Query(default="detailed", description="Display mode"),
    scope: Optional[str] = Query(None, description="Scope filter"),
    administration_ids: Optional[str] = Query(None, description="Administration IDs (comma-separated)"),
    department_ids: Optional[str] = Query(None, description="Department IDs (comma-separated)"),
    section_ids: Optional[str] = Query(None, description="Section IDs (comma-separated)"),
    include_charts: bool = Query(default=True, description="Include charts in export"),
    language: Literal["en", "ar"] = Query(default="en", description="Export language")
):
    """
    Export a monthly report.
    
    Returns the file content directly (not JSON metadata).
    Supports: pdf, csv, xlsx (Excel), docx (Word).
    """
    try:
        # Build filters from query parameters
        filters = {}
        if scope:
            filters["scope"] = scope
        if administration_ids:
            filters["administration_ids"] = administration_ids
        if department_ids:
            filters["department_ids"] = department_ids
        if section_ids:
            filters["section_ids"] = section_ids
        
        # Force report_type to monthly
        result = report_export_service.generate_export(
            report_type="monthly",
            display_mode=display_mode,
            file_format=format,
            year=year,
            month=month,
            trimester=None,
            quarter=None,
            filters=filters,
            include_charts=include_charts,
            language=language
        )
        
        # Return file directly instead of JSON metadata
        return Response(
            content=result["content"],
            media_type=result["content_type"],
            headers={
                "Content-Disposition": f"attachment; filename={result['filename']}"
            }
        )
    
    except Exception as e:
        # Log full exception with stack trace for debugging
        print("\n" + "="*80)
        print(f"[ROUTER] EXPORT HARD FAIL: {format.upper()} export")
        print(f"Parameters: year={year}, month={month}, display_mode={display_mode}")
        print(f"Exception: {type(e).__name__}: {str(e)}")
        print("="*80)
        traceback.print_exc()
        print("="*80 + "\n")
        
        # Emergency fallback for Word exports - NEVER return 500 for docx
        if format == "docx":
            print("[ROUTER] Attempting final emergency Word fallback...")
            try:
                if DOCX_AVAILABLE:
                    # Create absolute minimal Word document
                    doc = Document()
                    
                    # Just add text paragraphs - no tables, no styling
                    doc.add_paragraph("Emergency Fallback Word Export")
                    doc.add_paragraph()
                    doc.add_paragraph("The system encountered a critical error while generating your report.")
                    doc.add_paragraph()
                    doc.add_paragraph(f"Year: {year}")
                    doc.add_paragraph(f"Month: {month}")
                    doc.add_paragraph(f"Display Mode: {display_mode}")
                    doc.add_paragraph()
                    doc.add_paragraph(f"Error Type: {type(e).__name__}")
                    doc.add_paragraph(f"Error Details: {str(e)}")
                    doc.add_paragraph()
                    doc.add_paragraph("Please contact technical support with this information.")
                    
                    # Save to buffer
                    buffer = BytesIO()
                    doc.save(buffer)
                    buffer.seek(0)
                    
                    filename = f"Emergency_Export_{year}_{month:02d}.docx"
                    
                    print("[ROUTER] Emergency Word document created successfully - returning 200 OK")
                    return Response(
                        content=buffer.getvalue(),
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        headers={
                            "Content-Disposition": f"attachment; filename={filename}"
                        }
                    )
                else:
                    print("[ROUTER] python-docx not available - cannot create emergency fallback")
            
            except Exception as final_error:
                print(f"[ROUTER] Even final fallback failed: {final_error}")
                traceback.print_exc()
                # Absolutely last resort - return a text file renamed as docx
                try:
                    error_text = f"Emergency Export Failure\n\nYear: {year}\nMonth: {month}\n\nError: {str(e)}\n\nFinal Error: {str(final_error)}"
                    return Response(
                        content=error_text.encode('utf-8'),
                        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        headers={
                            "Content-Disposition": f"attachment; filename=Emergency_{year}_{month:02d}.docx"
                        }
                    )
                except:
                    pass  # Give up completely
        
        # For non-docx formats or if all fallbacks failed, return 500
        raise HTTPException(status_code=500, detail={
            "error": f"{format}_export_failed",
            "message": f"Failed to generate monthly {format.upper()}: {str(e)}",
            "message_ar": f"فشل إنشاء تقرير شهري {format.upper()}: {str(e)}",
            "exception_type": type(e).__name__,
            "exception_details": str(e)
        })
