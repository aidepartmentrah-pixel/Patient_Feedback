"""
Seasonal Comparison Routes
===========================
API endpoints for multi-quarter seasonal comparison reports.

Handles:
- 2-quarter comparison (current vs previous)
- 3-quarter comparison (with trend indicators)
- 4-quarter comparison (full year with yearly totals)
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import asyncio
import io

from backend.api.services.seasonal_comparison_service import seasonal_comparison_service
from backend.api.services.seasonal_report_formatter import (
    generate_comparative_seasonal_word_report,
    generate_3_quarter_comparison_report,
    generate_4_quarter_comparison_report
)

# Dedicated thread pool — heavy graph rendering and Word generation run here
# so the event loop stays responsive to other requests during generation.
_comparison_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="comparison-gen")


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class TwoQuarterComparisonRequest(BaseModel):
    """Request body for 2-quarter comparison"""
    season_ids: List[int] = Field(..., min_items=2, max_items=2, description="Exactly 2 season IDs (e.g., [4, 5] for Q4-2025 and Q1-2026)")
    orgunit_id: int = Field(..., ge=1, description="Organization unit ID")
    orgunit_type: int = Field(..., ge=0, le=3, description="Organization unit type (0=Hospital, 1=Administration, 2=Department, 3=Section)")
    user_id: int = Field(default=1, description="User ID requesting the report")
    format: str = Field(default="json", pattern="^(json|docx)$", description="Response format: 'json' for data, 'docx' for Word document download")


class ThreeQuarterComparisonRequest(BaseModel):
    """Request body for 3-quarter comparison"""
    season_ids: List[int] = Field(..., min_items=3, max_items=3, description="Exactly 3 consecutive season IDs")
    orgunit_id: int = Field(..., ge=1, description="Organization unit ID")
    orgunit_type: int = Field(..., ge=0, le=3, description="Organization unit type (0=Hospital, 1=Administration, 2=Department, 3=Section)")
    user_id: int = Field(default=1, description="User ID requesting the report")
    format: str = Field(default="json", pattern="^(json|docx)$", description="Response format: 'json' or 'docx'")


class FourQuarterComparisonRequest(BaseModel):
    """Request body for 4-quarter comparison (full year)"""
    season_ids: List[int] = Field(..., min_items=4, max_items=4, description="Exactly 4 consecutive season IDs (full year)")
    orgunit_id: int = Field(..., ge=1, description="Organization unit ID")
    orgunit_type: int = Field(..., ge=0, le=3, description="Organization unit type (0=Hospital, 1=Administration, 2=Department, 3=Section)")
    user_id: int = Field(default=1, description="User ID requesting the report")
    format: str = Field(default="json", pattern="^(json|docx)$", description="Response format: 'json' or 'docx'")


# ============================================================
# ROUTER
# ============================================================
router = APIRouter(prefix="/api/seasonal-comparison", tags=["Seasonal Comparison"])


# ============================================================
# 2-QUARTER COMPARISON ENDPOINT
# ============================================================

@router.post("/2-quarters", response_model=Dict[str, Any], status_code=200)
async def generate_two_quarter_comparison(
    request: TwoQuarterComparisonRequest
):
    """Generate a comparative report for 2 consecutive quarters."""
    try:
        loop = asyncio.get_event_loop()

        # Run DB fetch + graph generation off the event loop
        reports = await loop.run_in_executor(
            _comparison_executor,
            lambda: seasonal_comparison_service.fetch_multiple_seasonal_reports(
                season_ids=request.season_ids,
                orgunit_id=request.orgunit_id,
                orgunit_type=request.orgunit_type,
                user_id=request.user_id
            )
        )

        if len(reports) != 2:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 2 reports, got {len(reports)}. Check if season IDs exist."
            )

        changes = seasonal_comparison_service.calculate_percentage_changes(reports)
        periods = [report['header'].get('period', f'Q{i+1}') for i, report in enumerate(reports)]

        if request.format == "docx":
            doc_bytes = await loop.run_in_executor(
                _comparison_executor,
                lambda: generate_comparative_seasonal_word_report(
                    current_data=reports[1],
                    previous_data=reports[0]
                )
            )
            filename = f"2Quarter_Comparison_{periods[0]}_{periods[1]}.docx".replace(" ", "_")
            return StreamingResponse(
                io.BytesIO(doc_bytes),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        return {
            "success": True,
            "comparison_type": "2-quarters",
            "periods": periods,
            "season_ids": request.season_ids,
            "orgunit_id": request.orgunit_id,
            "orgunit_type": request.orgunit_type,
            "orgunit_name": reports[0]['header'].get('orgunit_name', 'Unknown'),
            "data": {
                "reports": reports,
                "percentage_changes": changes,
                "summary": {
                    "previous": {
                        "period": periods[0],
                        "total_cases": reports[0]['header'].get('total_cases', 0),
                        "clinical": reports[0]['header'].get('clinical_domain_count', 0),
                        "management": reports[0]['header'].get('management_domain_count', 0),
                        "relational": reports[0]['header'].get('relational_domain_count', 0)
                    },
                    "current": {
                        "period": periods[1],
                        "total_cases": reports[1]['header'].get('total_cases', 0),
                        "clinical": reports[1]['header'].get('clinical_domain_count', 0),
                        "management": reports[1]['header'].get('management_domain_count', 0),
                        "relational": reports[1]['header'].get('relational_domain_count', 0)
                    }
                }
            },
            "generated_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate 2-quarter comparison: {str(e)}"
        )


# ============================================================
# 3-QUARTER COMPARISON ENDPOINT
# ============================================================

@router.post("/3-quarters", response_model=Dict[str, Any], status_code=200)
async def generate_three_quarter_comparison(
    request: ThreeQuarterComparisonRequest
):
    """
    Generate a trend analysis report for 3 consecutive quarters.

    **Features:**
    - Trend indicators (↑↑, ↑, →, ↓, ↓↓) for all metrics
    - 3-column comparison tables (Q1 | Q2 | Q3 | Trend)
    - 3 spider chart visualizations (Domain, Category, SubCategory)
    - No bar charts or heatmaps

    **Request Body:**
    ```json
    {
        "season_ids": [4, 5, 6],
        "orgunit_id": 1,
        "orgunit_type": 0,
        "user_id": 1,
        "format": "json"  // or "docx"
    }
    ```
    
    **Response (format=json):**
    ```json
    {
        "success": true,
        "comparison_type": "3-quarters",
        "periods": ["Q4-2025", "Q1-2026", "Q2-2026"],
        "data": {
            "domain_comparison": {...},
            "category_comparison": {...},
            "trends": {...}
        }
    }
    ```
    """
    try:
        loop = asyncio.get_event_loop()

        # Run DB fetch off the event loop
        comparison_data = await loop.run_in_executor(
            _comparison_executor,
            lambda: seasonal_comparison_service.generate_3_quarter_comparison_data(
                season_ids=request.season_ids,
                orgunit_id=request.orgunit_id,
                orgunit_type=request.orgunit_type,
                user_id=request.user_id
            )
        )

        if request.format == "docx":
            # Run heavy Word + graph generation off the event loop
            def _build_3q_docx():
                doc = generate_3_quarter_comparison_report(comparison_data)
                buf = io.BytesIO()
                doc.save(buf)
                buf.seek(0)
                return buf.getvalue()

            doc_bytes = await loop.run_in_executor(_comparison_executor, _build_3q_docx)
            periods = comparison_data['periods']
            filename = f"3Quarter_Comparison_{periods[0]}_{periods[2]}.docx".replace(" ", "_")
            return StreamingResponse(
                io.BytesIO(doc_bytes),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        return {
            "success": True,
            "comparison_type": "3-quarters",
            "periods": comparison_data['periods'],
            "season_ids": comparison_data['season_ids'],
            "orgunit_id": comparison_data['orgunit_id'],
            "orgunit_type": comparison_data['orgunit_type'],
            "orgunit_name": comparison_data['orgunit_name'],
            "data": {
                "domain_comparison": comparison_data['domain_comparison'],
                "category_comparison": comparison_data['category_comparison'],
                "subcategory_comparison": comparison_data['subcategory_comparison'],
                "trends": comparison_data['trends']
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate 3-quarter comparison: {str(e)}"
        )


# ============================================================
# 4-QUARTER COMPARISON ENDPOINT
# ============================================================

@router.post("/4-quarters", response_model=Dict[str, Any], status_code=200)
async def generate_four_quarter_comparison(
    request: FourQuarterComparisonRequest
):
    """
    Generate a full-year annual report for 4 consecutive quarters.
    
    **Features:**
    - Yearly totals column (Q1 | Q2 | Q3 | Q4 | Yearly | Trend)
    - 4-series spider charts showing all quarters
    - Comprehensive annual analysis
    - Trend indicators for year-over-year comparison
    
    **Request Body:**
    ```json
    {
        "season_ids": [4, 5, 6, 7],
        "orgunit_id": 1,
        "orgunit_type": 0,
        "user_id": 1,
        "format": "json"  // or "docx"
    }
    ```
    
    **Response (format=json):**
    ```json
    {
        "success": true,
        "comparison_type": "4-quarters",
        "periods": ["Q4-2025", "Q1-2026", "Q2-2026", "Q3-2026"],
        "data": {
            "domain_comparison": {...},
            "yearly_totals": {...},
            "trends": {...}
        }
    }
    ```
    """
    try:
        loop = asyncio.get_event_loop()

        # Run DB fetch off the event loop
        comparison_data = await loop.run_in_executor(
            _comparison_executor,
            lambda: seasonal_comparison_service.generate_4_quarter_comparison_data(
                season_ids=request.season_ids,
                orgunit_id=request.orgunit_id,
                orgunit_type=request.orgunit_type,
                user_id=request.user_id
            )
        )

        if request.format == "docx":
            # Run heavy Word + graph generation off the event loop
            def _build_4q_docx():
                doc = generate_4_quarter_comparison_report(comparison_data)
                buf = io.BytesIO()
                doc.save(buf)
                buf.seek(0)
                return buf.getvalue()

            doc_bytes = await loop.run_in_executor(_comparison_executor, _build_4q_docx)
            periods = comparison_data['periods']
            filename = f"4Quarter_Annual_Report_{periods[0]}_{periods[3]}.docx".replace(" ", "_")
            return StreamingResponse(
                io.BytesIO(doc_bytes),
                media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )

        return {
            "success": True,
            "comparison_type": "4-quarters",
            "periods": comparison_data['periods'],
            "season_ids": comparison_data['season_ids'],
            "orgunit_id": comparison_data['orgunit_id'],
            "orgunit_type": comparison_data['orgunit_type'],
            "orgunit_name": comparison_data['orgunit_name'],
            "data": {
                "domain_comparison": comparison_data['domain_comparison'],
                "category_comparison": comparison_data['category_comparison'],
                "subcategory_comparison": comparison_data['subcategory_comparison'],
                "yearly_totals": comparison_data['yearly_totals'],
                "trends": comparison_data['trends']
            },
            "generated_at": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate 4-quarter comparison: {str(e)}"
        )


# ============================================================
# HELPER ENDPOINTS
# ============================================================

@router.get("/available-quarters", response_model=Dict[str, Any], status_code=200)
def get_available_quarters(
    orgunit_id: int = Query(..., ge=1, description="Organization unit ID"),
    orgunit_type: int = Query(..., ge=0, le=3, description="Organization unit type (0=Hospital, 1=Administration, 2=Department, 3=Section)")
):
    """
    Get list of available quarters/seasons for comparison.
    
    Useful for populating UI dropdowns or validating season_ids.
    """
    try:
        from backend.api.db_layer.seasonal_report import get_all_seasons
        
        seasons = get_all_seasons()
        
        return {
            "success": True,
            "orgunit_id": orgunit_id,
            "orgunit_type": orgunit_type,
            "available_seasons": [
                {
                    "season_id": s['SeasonID'],
                    "name": s['SeasonName'],
                    "start_date": s['StartDate'].isoformat() if s['StartDate'] else None,
                    "end_date": s['EndDate'].isoformat() if s['EndDate'] else None
                }
                for s in seasons
            ],
            "total_count": len(seasons)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch available quarters: {str(e)}"
        )
